import os
import json
import shutil
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from common.visualization import draw_map, draw_agent, draw_scen_trees, reset_ax, draw_traj_trees, draw_traj, draw_ghost_points
from common.geometry import get_vehicle_vertices, check_polygon_intersection

from agent import CustomizedAgent, NonReactiveAgent
from loader import ArgoAgentLoader
from common.semantic_map import SemanticMap
matplotlib.use('Agg')


class Simulator:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        self.sim_name = self.config['sim_name']
        self.seq_id = self.config['seq_id']
        self.output_dir = self.config['output_dir']
        self.num_threads = self.config['num_threads']
        self.seq_path = os.path.join('data/', self.seq_id)

        self.smp = SemanticMap()
        self.smp.load_from_argo2(Path(self.seq_path + f"/log_map_archive_{self.seq_id}.json"))

        self.render = self.config['render']
        self.cl_agents = self.config['cl_agents']

        self.sim_time = 0.0
        self.sim_step = 0.02
        self.sim_horizon = 500
        self.agents = []
        self.frames = []
        self.collision_log = [] # 记录碰撞数据

    def run(self):
        self.init_sim()
        self.run_sim()
        self.render_video()
        self.save_collision_report()

    def init_sim(self):
        self.agents = []
        scenario_path = Path(self.seq_path + f"/scenario_{self.seq_id}.parquet")
        replay_agent_loader = ArgoAgentLoader(scenario_path)
        self.agents += replay_agent_loader.load_agents(self.smp, self.cl_agents)
        self.collision_log = []

    def get_agent_polygon(self, agent):
        # 简化：使用 3D 框的底部 4 个点投影到 2D
        # agent.state: [x, y, v, heading]
        # agent.bbox: l, w, h
        x, y, _, heading = agent.state
        l, w, h = agent.bbox.l, agent.bbox.w, agent.bbox.h
        
        # 假设地面 z=0
        vertices_3d = get_vehicle_vertices(x, y, 0, heading, l, w, h)
        # 取底面 4 个点 (前4个)
        poly_2d = [v[:2] for v in vertices_3d[:4]]
        return np.array(poly_2d)

    def run_sim(self):
        print("Running simulation...")
        # reset sim time and frames
        self.frames = []
        self.sim_time = 0.0
        terminated = False
        collided = False

        for step_idx in tqdm(range(self.sim_horizon)):
            frame = {}
            # Update agent observations
            agent_obs = []
            for agent in self.agents:
                if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                    agent_obs.append(agent.observe())

            # Record ground truth
            agent_gt = []
            for agent in self.agents:
                if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                    agent_gt.append(agent.observe_no_noise())

            frame['agents'] = agent_gt
            
            # --- Collision Check ---
            # 找到 Ego
            ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
            if ego_agent and ego_agent.is_enable and not collided:
                ego_poly = self.get_agent_polygon(ego_agent)
                
                for other in self.agents:
                    if other.id == 'AV': continue
                    # 距离预筛选 (例如 10m 内)
                    if np.linalg.norm(other.state[:2] - ego_agent.state[:2]) > 10.0:
                        continue
                        
                    other_poly = self.get_agent_polygon(other)
                    if check_polygon_intersection(ego_poly, other_poly):
                        print(f"\n[COLLISION] At {self.sim_time:.2f}s: Ego collided with {other.id} ({other.type})")
                        collided = True # 标记为已碰撞，避免重复记录同一事故
                        self.collision_log.append({
                            "timestamp": float(self.sim_time),
                            "frame_idx": step_idx,
                            "ego_state": ego_agent.state.tolist(), # [x,y,v,h]
                            "ego_vel": float(ego_agent.state[2]),
                            "other_id": str(other.id),
                            "other_type": str(other.type),
                            "other_state": other.state.tolist(),
                            "collision_msg": f"Collision with {other.type} ID:{other.id}"
                        })
            
            # Update local semantic map and plan
            for agent in self.agents:
                if isinstance(agent, CustomizedAgent):
                    agent.check_enable(self.sim_time)
                    rec_tri, pl_tri = agent.check_trigger(self.sim_time)

                    if rec_tri:
                        agent.step()
                    if pl_tri:
                        agent.update_observation(agent_obs)
                        if agent.is_enable:  # if enable then plan to get control
                            is_success, res = agent.plan()
                            if not is_success:
                                print("Agent {} plan failed!".format(agent.id))
                                terminated = True
                                break

                            # hack for recording the planning result
                            if agent.id == 'AV':
                                frame['scen_tree'] = res[0]
                                frame['traj_tree'] = res[1]
                                # 保存鬼探头点用于可视化
                                if len(res) > 2:
                                    frame['ghost_points'] = res[2]
                                # 保存目标车道
                                if hasattr(agent, 'gt_tgt_lane'):
                                    frame['target_lane'] = agent.gt_tgt_lane

                elif isinstance(agent, NonReactiveAgent):
                    agent.step()
                else:
                    raise ValueError("Unknown agent type")
                agent.update_state(self.sim_step)

            self.frames.append(frame)
            
            # [MEMORY FIX] 如果不渲染，定期清理 frame 缓存，防止内存爆炸
            if not self.render and len(self.frames) > 50:
                self.frames = self.frames[-10:]
                
            self.sim_time += self.sim_step

            if terminated:
                print("Simulation terminated!")
                break
                
    def save_collision_report(self):
        # 按照用户要求，把报告放到 output/xxx/imgs/ 文件夹旁边，或者直接放到 imgs 里
        # 但通常 imgs 里全是图片，放在同级的 sim_name 文件夹下比较好
        # 用户需求："放到跟图片一起的文件夹里面" -> 这里理解为 output_dir 下 (也就是 imgs 的父目录)
        # 或者为了方便，我们直接放到 imgs 里面也行? 
        # 用户原话："把那个碰撞报告也放到跟图片一起的文件夹里面"
        
        img_dir = self.output_dir + '/imgs'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            
        report_path = os.path.join(img_dir, 'collision_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(self.collision_log, f, indent=2)
        
        if self.collision_log:
            print(f"⚠️  Collision data saved to {report_path} ({len(self.collision_log)} events)")
        else:
            print(f"✅  No collisions detected. Report saved to {report_path}")

    def render_video(self):
        if not self.render:
            return
        print("Rendering video...")
        # check directory exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        img_dir = self.output_dir + '/imgs'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # Single-threaded rendering to avoid macOS multiprocessing issues
        from tqdm import tqdm
        for frame_idx in tqdm(range(len(self.frames)), desc="Rendering frames"):
            self.render_png(frame_idx, img_dir)

        # call ffmpeg to combine images into a video
        video_name = f'{self.seq_id}_{self.sim_name}.mov'
        output_command = "ffmpeg -r 25 -i " + img_dir + f'/frame_%03d.png' + " -vcodec mpeg4 -y " + \
                         self.output_dir + video_name
        os.system(output_command)
        # Keep the images for comparison (disabled deletion)
        # shutil.rmtree(img_dir)
        print(f"Images saved to: {img_dir}")

    ########################################
    # Visualization functions
    ########################################
    def render_png(self, frame_idx, img_dir):
        fig = plt.figure(figsize=(48, 48))
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()
        self.render_frame(frame_idx, ax)
        # Save the frame with directory path
        frame_filename = img_dir + f'/frame_{frame_idx:03d}.png'
        plt.tight_layout()
        plt.savefig(frame_filename)
        plt.close(fig)
#这个函数负责画图
    def render_frame(self, frame_idx, ax):
        scen_tree_vis = None
        traj_tree_vis = None

        # retrieve the vis data from the previous frame to avoid the empty visualization
        if 'scen_tree' in self.frames[frame_idx]:
            scen_tree_vis = self.frames[frame_idx]['scen_tree']
        else:
            pre_frame_idx = frame_idx - 1
            while pre_frame_idx >= 0 and 'scen_tree' not in self.frames[pre_frame_idx]:
                pre_frame_idx -= 1
            if pre_frame_idx >= 0 and 'scen_tree' in self.frames[pre_frame_idx]:
                scen_tree_vis = self.frames[pre_frame_idx]['scen_tree']

        if 'traj_tree' in self.frames[frame_idx]:
            traj_tree_vis = self.frames[frame_idx]['traj_tree']
        else:
            pre_frame_idx = frame_idx - 1
            while pre_frame_idx >= 0 and 'traj_tree' not in self.frames[pre_frame_idx]:
                pre_frame_idx -= 1
            if pre_frame_idx >= 0 and 'traj_tree' in self.frames[pre_frame_idx]:
                traj_tree_vis = self.frames[pre_frame_idx]['traj_tree']

        # Clear the previous cube and draw a new one
        range_3d = 15.0
        font_size = 35
        reset_ax(ax)

        # Process the frame
        center = np.array([0.0, 0.0])
        cam_yaw = self.config['render_config']['camera_position']['yaw']
        elev = self.config['render_config']['camera_position']['elev']

        # Handle 'follow' mode
        if self.config['render_config'].get('mode') == 'follow':
            # Find AV agent
            for agent in self.frames[frame_idx]['agents']:
                if agent.id == 'AV':  # Assuming AV ID is always 'AV'
                    center[0] = agent.state[0]
                    center[1] = agent.state[1]
                    # Optional: Lock yaw to agent heading
                    # cam_yaw = agent.state[3]
                    break
        else:
            # Fixed mode
            center[0] = self.config['render_config']['camera_position']['x']
            center[1] = self.config['render_config']['camera_position']['y']

        ax.set_xlim([center[0] - range_3d, center[0] + range_3d])
        ax.set_ylim([center[1] - range_3d, center[1] + range_3d])
        ax.set_zlim([0, 2 * range_3d])
        ax.view_init(elev=elev, azim=180 + np.rad2deg(cam_yaw))

        draw_map(ax, self.smp.map_data)
        if scen_tree_vis is not None:
            draw_scen_trees(ax, scen_tree_vis)
        
        # 绘制 Ground Truth (标准答案) 轨迹 (红色细线)
        # 不再画 target_lane，而是画 AV Agent 的原始轨迹
        av_agent = next((a for a in self.agents if a.id == 'AV'), None)
        if av_agent and hasattr(av_agent, 'traj_info'):
            # traj_info[0] is ALREADY the position array [N, 2]
            gt_traj = av_agent.traj_info[0]
            # Remove zeros or invalid points if necessary, but usually full traj is fine
            from common.visualization import draw_polyline
            draw_polyline(ax, gt_traj, z=0.05, width=0.5, color='red')
                
        if traj_tree_vis is not None:
            draw_traj_trees(ax, traj_tree_vis)
        
        # 绘制鬼探头危险区域
        ghost_points_vis = None
        if 'ghost_points' in self.frames[frame_idx]:
            ghost_points_vis = self.frames[frame_idx]['ghost_points']
        else:
            # 从之前的帧获取
            pre_frame_idx = frame_idx - 1
            while pre_frame_idx >= 0 and 'ghost_points' not in self.frames[pre_frame_idx]:
                pre_frame_idx -= 1
            if pre_frame_idx >= 0 and 'ghost_points' in self.frames[pre_frame_idx]:
                ghost_points_vis = self.frames[pre_frame_idx]['ghost_points']
        
        if ghost_points_vis is not None and len(ghost_points_vis) > 0:
            draw_ghost_points(ax, ghost_points_vis)

        #  plot agents
        for agent in self.frames[frame_idx]['agents']:
            draw_agent(ax, agent, vis_bbox=False)
            if np.linalg.norm(agent.state[:2] - center) < 2 * range_3d:
                ax.text(agent.state[0], agent.state[1], 1.0, 'No.{}:{:.2f}m/s'.format(agent.id, agent.state[2]),
                           fontsize=font_size)

        # try to retrieve the history of the agent in current frame
        agent_history = dict()
        for agent in self.frames[frame_idx]['agents']:
            agent_history[agent.id] = [agent.state[:2]]

        back_step = 100
        for i in range(1, back_step):
            if frame_idx - i < 0:
                break
            for agent in self.frames[frame_idx - i]['agents']:
                if agent.id in agent_history:
                    agent_history[agent.id].append(agent.state[:2])

        # plot the history of the agent
        for agent_id, history in agent_history.items():
            history.reverse()
            # check length of history
            if np.linalg.norm(history[0] - history[-1]) < 0.1:
                continue
            draw_traj(ax, history)
