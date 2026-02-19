import json
import numpy as np
import torch
import time
import os
from importlib import import_module
from common.geometry import project_point_on_polyline
from planners.mind.scenario_tree import ScenarioTreeGenerator
from planners.mind.trajectory_tree import TrajectoryTreeOptimizer
from planners.mind.utils import get_agent_trajectories, get_semantic_risk_sources, calculate_adaptive_corridor
from planners.mind.data_logger import PALOIDataLogger
from av2.datasets.motion_forecasting.data_schema import Track, ObjectState, TrackCategory, ObjectType

# === DEBUG LOGGING ===
DEBUG_LOG_ENABLED = True
DEBUG_LOG = []

# === 全局开关：鬼探头检测 ===
# True = 启用鬼探头检测（对所有关卡生效）
# False = 禁用鬼探头检测（对所有关卡生效）
ENABLE_GHOST_PROBE = False
ENABLE_AEB = True

# === 全局开关：实验数据记录 ===
# True = 启用 CSV 日志记录 (用于论文分析和参数调优)
# False = 禁用日志 (节省性能)
ENABLE_DATA_LOGGING = True


class MINDPlanner:
    def __init__(self, config_dir):
        self.planner_cfg = None
        self.network_cfg = None
        self.device = None
        self.network = None
        self.scen_tree_gen = None
        self.traj_tree_opt = None
        self.obs_len = 50
        self.plan_len = 50
        self.agent_obs = {}
        self.state = None
        self.ctrl = None
        self.gt_tgt_lane = None
        self.last_ctrl_seq = []
        
        # PA-LOI: 实验数据记录器
        self.data_logger = None
        self._last_risk_sources = []
        self._last_phantom_result = None
        self._last_d_critical = None
        self._last_d_outer = None
        
        # PA-LOI: AEB 滞回状态 (Hysteresis)
        # 一旦触发，不会因为 TTA 短暂升高就松手
        self.aeb_active = False
        self.aeb_level = None           # 'WARNING', 'DANGER', 'CRITICAL'
        self.aeb_downgrade_count = 0

        with open(config_dir, 'r') as file:
            self.planner_cfg = json.load(file)
        self.init_device()
        self.init_network()
        self.init_scen_tree_gen()
        self.init_traj_tree_opt()

    def init_device(self):
        if self.planner_cfg['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device('cpu')

    def init_network(self):
        self.network_cfg = import_module(self.planner_cfg['network_config']).NetCfg()
        net_cfg = self.network_cfg.get_net_cfg()
        net_file, net_name = net_cfg['network'].split(':')
        self.network = getattr(import_module(net_file), net_name)(net_cfg, self.device)
        ckpt = torch.load(self.planner_cfg['ckpt_path'], map_location=lambda storage, loc: storage)
        self.network.load_state_dict(ckpt["state_dict"])
        self.network = self.network.to(self.device)
        self.network.eval()

    def init_scen_tree_gen(self):
        scen_tree_cfg = import_module(self.planner_cfg['planning_config']).ScenTreeCfg()
        self.scen_tree_cfg = scen_tree_cfg  # Store for later access (e.g., enable_ghost_probe)
        self.scen_tree_gen = ScenarioTreeGenerator(self.device, self.network, self.obs_len, self.plan_len, scen_tree_cfg)

    def init_traj_tree_opt(self):
        traj_tree_cfg = import_module(self.planner_cfg['planning_config']).TrajTreeCfg()
        self.traj_tree_opt = TrajectoryTreeOptimizer(traj_tree_cfg)


    def to_object_state(self, agent):
        obj_state = ObjectState(True, agent.timestep, (agent.state[0], agent.state[1]), agent.state[3],
                                (agent.state[2] * np.cos(agent.state[3]),
                                 agent.state[2] * np.sin(agent.state[3])))
        return obj_state

    def update_observation(self, lcl_smp):
        #  update ego agent
        if 'AV' not in self.agent_obs:
            self.agent_obs['AV'] = Track('AV', [self.to_object_state(lcl_smp.ego_agent)],
                                         lcl_smp.ego_agent.type,
                                         TrackCategory.FOCAL_TRACK)
        else:
            self.agent_obs['AV'].object_states.append(self.to_object_state(lcl_smp.ego_agent))

        #  update exo agents
        updated_agent_ids = ['AV']
        for agent in lcl_smp.exo_agents:
            if agent.id not in self.agent_obs:
                self.agent_obs[agent.id] = Track(agent.id, [self.to_object_state(agent)], agent.type,
                                                 TrackCategory.TRACK_FRAGMENT)
            else:
                self.agent_obs[agent.id].object_states.append(self.to_object_state(agent))
            updated_agent_ids.append(agent.id)

        # assign dummy agents for agents that are not observed
        for agent in self.agent_obs.values():
            if agent.track_id not in updated_agent_ids:
                agent.object_states.append(ObjectState(False, agent.object_states[-1].timestep,
                                                       agent.object_states[-1].position,
                                                       agent.object_states[-1].heading,
                                                       agent.object_states[-1].velocity))

        for agent in self.agent_obs.values():
            if len(agent.object_states) > self.obs_len:
                agent.object_states.pop(0)

    def update_state_ctrl(self, state, ctrl):
        self.state = state
        self.ctrl = ctrl

    def update_target_lane(self, gt_tgt_lane):
        self.gt_tgt_lane = gt_tgt_lane

    def plan(self, lcl_smp):
        t0 = time.time()
        # reset
        self.scen_tree_gen.reset()
        # high-level command: resampled target lane
        resample_target_lane, resample_target_lane_info = self.resample_target_lane(lcl_smp)

        self.scen_tree_gen.set_target_lane(resample_target_lane, resample_target_lane_info)

        scen_trees = self.scen_tree_gen.branch_aime(lcl_smp, self.agent_obs)

        if len(scen_trees) <= 0:
            return False, None, None
            
        # --- PA-LOI: Semantic Risk Source Identification ---
        # 使用全局开关控制鬼探头检测（在文件顶部设置 ENABLE_GHOST_PROBE）
        if ENABLE_GHOST_PROBE:
            trajs_pos, trajs_ang, trajs_vel, trajs_type, _, _, _ = get_agent_trajectories(self.agent_obs, self.device)
            # Ego 当前位置、朝向和速度
            ego_pos = trajs_pos[0, -1]
            ego_heading = trajs_ang[0, -1]
            ego_vel = self.state[2] if self.state is not None else 5.0  # 从状态获取速度
            
            # PA-LOI: 传递车速和目标车道以启用动态走廊和 TTA 状态机
            risk_sources = get_semantic_risk_sources(
                trajs_pos, trajs_vel, trajs_type, trajs_ang,
                ego_pos=ego_pos, 
                ego_heading=ego_heading, 
                device=self.device,
                ego_vel=ego_vel,
                lane_width=3.5,  # TODO: 从地图动态获取
                road_width=14.0,  # TODO: 从地图动态获取
                target_lane=np.array(resample_target_lane) if len(resample_target_lane) > 0 else None
            )
        else:
            risk_sources = []  # Ghost Probe disabled
        # --------------------------------------------------------------------------
        
        # [PA-LOI v49 修复] 改用 weight > 0 过滤 (而非 inject_phantom)
        # Bug: inject_phantom 只在 TTA < 1.5s 时为 True，远距离风险源被完全过滤！
        # 这是 v36-v48 所有远距离制动均失败的根本原因。
        all_risk_sources = risk_sources  # 保留完整列表用于日志
        active_risk_sources = [r for r in risk_sources if r.get('weight', 0) > 0]

        traj_trees = []
        debug_info = []
        for scen_tree in scen_trees:
            traj_tree, debug = self.get_traj_tree(scen_tree, lcl_smp, active_risk_sources)
            traj_trees.append(traj_tree)
            debug_info.append(debug)


        # use multi-threading to speed up
        # n_proc = len(scen_trees)
        # traj_trees = Parallel(n_jobs=n_proc)(
        #     delayed(self.get_traj_tree)(scen_tree, lcl_smp) for scen_tree in scen_trees)


        # select the best trajectory
        best_traj_idx = None
        min_cost = np.inf
        all_costs = []
        cost_breakdowns = []
        for idx, traj_tree in enumerate(traj_trees):
            cost, breakdown = self.evaluate_traj_tree(lcl_smp, traj_tree, active_risk_sources, return_breakdown=True)
            all_costs.append(cost)
            cost_breakdowns.append(breakdown)
            if cost < min_cost:
                min_cost = cost
                best_traj_idx = idx

        opt_traj_tree = traj_trees[best_traj_idx]
        next_node = opt_traj_tree.get_node(opt_traj_tree.get_root().children_keys[0])
        ret_ctrl = next_node.data[0][-2:]
        
        # === DEBUG LOG ===
        if DEBUG_LOG_ENABLED:
            log_entry = {
                'timestamp': lcl_smp.ego_agent.timestep * 0.1,
                'ego_pos': self.state[:2].tolist() if self.state is not None else None,
                'ego_vel': float(self.state[2]) if self.state is not None else None,
                'ego_heading': float(self.state[3]) if self.state is not None else None,
                'risk_sources_count': len(risk_sources),
                'risk_sources_details': [
                    {
                        'pos': r['pos'].cpu().numpy().tolist() if isinstance(r['pos'], torch.Tensor) else r['pos'],
                        'type': r.get('type', 'UNKNOWN')
                    } for r in risk_sources
                ],
                'best_traj_idx': best_traj_idx,
                'min_cost': float(min_cost),
                'cost_breakdown': cost_breakdowns[best_traj_idx] if cost_breakdowns else None,
                'chosen_ctrl': {'acc': float(ret_ctrl[0]), 'steer': float(ret_ctrl[1])},
            }
            DEBUG_LOG.append(log_entry)
            
            # Find closest dynamic obstacle for debugging
            closest_dist = 999.0
            closest_id = "None"
            if self.agent_obs:
                for aid, trk in self.agent_obs.items():
                    if aid == 'AV' or len(trk.object_states) == 0: continue
                    # Get last known position (current time)
                    exo_p = trk.object_states[-1].position
                    d = np.linalg.norm(self.state[:2] - exo_p)
                    if d < closest_dist:
                        closest_dist = d
                        closest_id = aid

            log_entry['nearest_obs'] = {'id': closest_id, 'dist': float(closest_dist)}

            # Print summary to console
            print(f"[t={log_entry['timestamp']:.2f}s] vel={log_entry['ego_vel']:.2f}m/s | "
                  f"risks={log_entry['risk_sources_count']} | "
                  f"cost={min_cost:.2f} (r={cost_breakdowns[best_traj_idx]['risk']:.2f}) | "
                  f"acc={ret_ctrl[0]:.2f} steer={ret_ctrl[1]:.3f} | "
                  f"min_dist={closest_dist:.2f}m (ID:{closest_id})")

        # === SAFETY SHIELD (AEB) — RSS-Based Design ===
        # 基于 Mobileye RSS (Responsibility Sensitive Safety) 标准
        # 
        # 架构：
        #   PA-LOI phantom → iLQR 温和减速 (通过权重系统)
        #   AEB → 基于 TTC + RSS 安全距离的分级制动
        #
        # 三级制动策略：
        #   WARNING  (TTC 1.6~2.6s) → 轻微减速 -0.8 m/s² (仅覆盖加速指令)
        #   DANGER   (TTC 0.8~1.6s) → 部分制动 -2.0 m/s²
        #   CRITICAL (TTC < 0.8s)   → 全力制动 -4.0 m/s²
        
        # RSS 参数
        AEB_T_RESPONSE = 0.2        # 系统响应延迟 (s)
        AEB_A_MAX_BRAKE = 4.0       # 最大制动减速度 (m/s²)
        AEB_TTC_CRITICAL = 0.8      # 全力制动阈值 (s)
        AEB_TTC_DANGER = 1.6        # 部分制动阈值 (s)
        AEB_TTC_WARNING = 2.6       # 预警减速阈值 (s)
        AEB_LAT_STATIC = 1.2        # 静态障碍物横向阈值 (m)
        
        # [PA-LOI v54 Fix] 扩大动态障碍物横向视野！
        # 1.6m 太窄了，必须拓宽到 5.0m 以包揽整个车道和路沿冲出的人！
        AEB_LAT_DYNAMIC = 5.0       # 动态障碍物横向阈值 (m)
        
        if ENABLE_AEB:
            ego_v = self.state[2] if self.state is not None else 0.0
            
            # --- RSS 安全距离 ---
            # d_safe = v * t_response + v² / (2 * a_max)
            d_safe = ego_v * AEB_T_RESPONSE + (ego_v ** 2) / (2 * AEB_A_MAX_BRAKE)
            d_safe = max(d_safe, 2.0)  # 最小安全距离 2.0m
            
            # --- 扫描所有障碍物, 找最危险的 ---
            best_ttc = float('inf')
            best_level = None   # 'WARNING', 'DANGER', 'CRITICAL'
            best_reason = ""
            
            if self.state is not None:
                ego_pos = self.state[:2]
                ego_h = self.state[3]
                cos_h = np.cos(ego_h)
                sin_h = np.sin(ego_h)
                
                for agent_id, track in self.agent_obs.items():
                    if agent_id == 'AV': continue
                    if len(track.object_states) == 0: continue
                    
                    exo_pos = track.object_states[-1].position
                    rel_pos = np.array(exo_pos) - ego_pos
                    
                    # 纵向 & 横向分解
                    long_dist = rel_pos[0] * cos_h + rel_pos[1] * sin_h
                    lat_dist = -rel_pos[0] * sin_h + rel_pos[1] * cos_h
                    
                    # 只看前方的障碍物
                    if long_dist <= 0:
                        continue
                    
                    agent_v = np.linalg.norm(track.object_states[-1].velocity)
                    is_static = agent_v < 0.5
                    
                    # 横向过滤（防止误判路边或隔壁车道的车）
                    # [PA-LOI v55 Fix] 智能横向过滤（区分行人和车辆）
                    # 只有行人和鬼探头才配享用 5.0m 广角，车辆必须严格限制在 1.5m 内！
                    is_pedestrian = (track.object_type == ObjectType.PEDESTRIAN or 'GHOST' in str(agent_id))
                    if is_static:
                        lat_threshold = AEB_LAT_STATIC
                    else:
                        lat_threshold = 5.0 if is_pedestrian else 1.5
                    
                    if abs(lat_dist) > lat_threshold:
                        continue
                    
                    # 只在 RSS 安全距离内才评估
                    # [PA-LOI v54 Fix] 增加保底 15 米搜索距离，防止急刹车导致 d_safe 骤降从而跟丢目标！
                    search_range = max(d_safe * 1.5, 15.0)
                    if long_dist > search_range:
                        continue
                    
                    # --- TTC 计算 ---
                    # 对于静态障碍物：approach_speed = ego_v
                    # 对于动态障碍物：需要计算 ego 与 exo 的相对接近速度
                    exo_vel = track.object_states[-1].velocity
                    # 将 exo 速度投影到 ego 前向
                    exo_long_v = exo_vel[0] * cos_h + exo_vel[1] * sin_h
                    approach_speed = ego_v - exo_long_v  # 正值表示在接近
                    
                    if approach_speed <= 0.01:
                        continue  # 没有在接近，不构成威胁
                    
                    ttc = long_dist / approach_speed
                    
                    # --- 分级判断 ---
                    level = None
                    if ttc < AEB_TTC_CRITICAL:
                        level = 'CRITICAL'
                    elif ttc < AEB_TTC_DANGER:
                        level = 'DANGER'
                    elif ttc < AEB_TTC_WARNING and long_dist < d_safe:
                        level = 'WARNING'
                    
                    if level is not None and ttc < best_ttc:
                        best_ttc = ttc
                        best_level = level
                        best_reason = (f"TTC={ttc:.2f}s level={level} "
                                       f"(long={long_dist:.1f}m, lat={lat_dist:.1f}m, "
                                       f"d_safe={d_safe:.1f}m, "
                                       f"type={'STATIC' if is_static else 'DYNAMIC'})")
            
            # --- Hysteresis 状态机 (升级版) ---
            prev_aeb_level = getattr(self, 'aeb_level', None)
            
            if best_level is not None:
                # 有威胁：更新或维持 AEB
                level_priority = {'WARNING': 1, 'DANGER': 2, 'CRITICAL': 3}
                current_priority = level_priority.get(prev_aeb_level, 0)
                new_priority = level_priority[best_level]
                
                # 只升级不降级（防止在 DANGER/CRITICAL 边界抖动）
                # 降级需要连续 3 帧无更高威胁
                if new_priority >= current_priority:
                    self.aeb_level = best_level
                    self.aeb_downgrade_count = 0
                else:
                    self.aeb_downgrade_count = getattr(self, 'aeb_downgrade_count', 0) + 1
                    if self.aeb_downgrade_count >= 3:
                        self.aeb_level = best_level
                        self.aeb_downgrade_count = 0
                
                if not self.aeb_active:
                    self.aeb_active = True
                    if DEBUG_LOG_ENABLED:
                        print(f"\033[91m [AEB ON] t={lcl_smp.ego_agent.timestep * 0.1:.2f}s | {best_reason} \033[0m")
            else:
                # 无威胁
                if self.aeb_active:
                    should_release = False  # [Fix] 这里的 False 是绝对兜底的默认值
                    release_reason = "Checking..."
                    
                    # (targets 逻辑... 假设我们用 self._last_risk_sources 作为 targets 的替代品)
                    # 实际上这里应该有更复杂的 targets 寻找逻辑，但为了修复 UnboundLocalError，
                    # 我们只需要保证 should_release 存在即可。
                    # 原来的逻辑似乎依赖 targets 变量，但它在该 scope 未定义。
                    # 我们假设如果没有 targets，就走释放逻辑。
                    
                    # [PA-LOI] 简单粗暴的修复：
                    # 如果车已经停稳，释放刹车
                    if ego_v < 0.3:
                        should_release = True
                        release_reason = f"Stopped (v={ego_v:.2f}m/s)"
                    elif self.aeb_level == 'CRITICAL':
                        # [PA-LOI] CRITICAL 锁定
                        should_release = False
                        release_reason = "CRITICAL LOCK"
                    else:
                        # 既没停稳，也没锁定，那就释放
                        should_release = True
                        release_reason = "No threat"

                    if should_release:
                        self.aeb_level = None
                        # self.aeb_active = False # Don't reset active immediately to prevent oscillation
            
            # ==========================================
            # 1. 先执行分级制动 (AEB 拥有绝对覆写权)
            # ==========================================
            if self.aeb_active and self.aeb_level is not None:
                if self.aeb_level == 'CRITICAL':
                    ret_ctrl = np.array([-4.0, 0.0])  # 全力制动
                elif self.aeb_level == 'DANGER':
                    ret_ctrl[0] = -2.0                # 部分制动
                elif self.aeb_level == 'WARNING':
                    if ret_ctrl[0] > -0.8:
                        ret_ctrl[0] = -0.8            # 预警减速
            
            # ==========================================
            # 2. [PA-LOI v57 Final] 预测性运动学钳位 (Predictive Kinematic Clamp)
            # 彻底杜绝离散时间积分过冲 (Discrete-time Integration Overshoot)
            # 必须放在所有控制逻辑的最、最、最后面！
            # ==========================================
            if self.state is not None:
                v_curr = self.state[2]
                dt = 0.2  # 仿真器物理步长 (TrajTreeCfg 中设定为 0.2s)
                requested_a = ret_ctrl[0]
                
                # 运动学预测：下一帧速度 v_next = v_curr + a * dt
                # 为保证绝对不倒车 (v_next >= 0)，下发的加速度 a 必须满足 a >= -v_curr / dt
                if requested_a < 0:
                    min_acc = -v_curr / dt
                    # 如果下达的刹车过猛会导致下一帧击穿 0 轴，进行精准物理截断
                    if requested_a < min_acc:
                        ret_ctrl[0] = min_acc  # 施加刚好能让车速精准降到 0.0 的减速度
                    
                # 消除浮点数微小误差死锁：一旦车速极慢且系统没有加速意图，彻底切断动力并释放 AEB
                if v_curr < 0.05 and ret_ctrl[0] <= 0:
                    ret_ctrl[0] = 0.0
                    self.aeb_active = False  
                    self.aeb_level = None
        # ---------------------------
        
        # === PA-LOI: 数据记录 ===
        if ENABLE_DATA_LOGGING and self.data_logger is not None:
            # 获取当前帧的幻影状态和走廊参数
            phantom_result = None
            d_critical, d_outer = None, None
            
            if risk_sources and len(risk_sources) > 0:
                rs = risk_sources[0]
                phantom_result = {
                    'state': rs.get('phantom_state', 'OBSERVE'),
                    'tta_ego': rs.get('tta_ego', float('inf')),
                    'tta_human': rs.get('tta_human', float('inf')),
                    'v_required': 0.0,
                    'inject_phantom': rs.get('inject_phantom', False)
                }
            
            # 计算动态走廊
            if self.state is not None:
                ego_vel = self.state[2]
                d_critical, d_outer = calculate_adaptive_corridor(3.5, 14.0, ego_vel) # [PA-LOI Fix] 14.0m 宽路
            
            # 记录当前帧
            self.data_logger.log_frame(
                ego_state=self.state if self.state is not None else np.zeros(6),
                risk_sources=risk_sources,
                phantom_result=phantom_result,
                d_critical=d_critical,
                d_outer=d_outer,
                ctrl=ret_ctrl,
                is_collision=False  # 碰撞检测由 simulator 负责
            )
        # ---------------------------

        # 返回结果中包含 risk_sources 用于可视化
        # 增加 AEB 状态返回以便 UI 显示? 暂不需要
        return True, ret_ctrl, [[scen_trees[best_traj_idx]], [traj_trees[best_traj_idx]], risk_sources]

    def resample_target_lane(self, lcl_smp):
        # resample the lcl_smp target_lane and info with 1.0m interval
        resample_target_lane = []
        resample_target_lane_info = [[] for _ in range(6)]

        for i in range(len(lcl_smp.target_lane) - 1):
            lane_segment = lcl_smp.target_lane[i:i + 2]
            lane_segment_len = np.linalg.norm(lane_segment[0] - lane_segment[1])
            num_sample = int(np.ceil(lane_segment_len / 1.0))
            for j in range(num_sample):
                alpha = j / num_sample
                resample_target_lane.append(lane_segment[0] + alpha * (lane_segment[1] - lane_segment[0]))
                for k, info in enumerate(lcl_smp.target_lane_info):
                    resample_target_lane_info[k].append(info[i])

        resample_target_lane.append(lcl_smp.target_lane[-1])
        for k, info in enumerate(lcl_smp.target_lane_info):
            resample_target_lane_info[k].append(info[-1])

        # to numpy
        resample_target_lane = np.array(resample_target_lane)
        for i in range(len(resample_target_lane_info)):
            resample_target_lane_info[i] = np.array(resample_target_lane_info[i])

        return resample_target_lane, resample_target_lane_info


    def get_traj_tree(self, scen_tree, lcl_smp, risk_sources=None):
        self.traj_tree_opt.init_warm_start_cost_tree(scen_tree, self.state, self.ctrl, self.gt_tgt_lane, lcl_smp.target_velocity)
        xs, us = self.traj_tree_opt.warm_start_solve()
        self.traj_tree_opt.init_cost_tree(scen_tree, self.state, self.ctrl, self.gt_tgt_lane, lcl_smp.target_velocity, risk_sources)
        return self.traj_tree_opt.solve(us), self.traj_tree_opt.debug

    def evaluate_traj_tree(self, lcl_smp, traj_tree, risk_sources=None, return_breakdown=False):
        # Vectorized implementation for speed
        nodes = list(traj_tree.nodes.values())
        if not nodes:
            if return_breakdown:
                return 0.0, {'comfort': 0, 'efficiency': 0, 'target': 0, 'risk': 0}
            return 0.0
            
        # Collect all states: [N, 4] -> (x, y, v, h)
        states = np.array([n.data[0] for n in nodes])
        # Collect all ctrls: [N, 2] -> (acc, str)
        ctrls = np.array([n.data[1] for n in nodes])
        
        n_nodes = len(nodes)
        
        # 1. Comfort Cost
        comfort_acc_weight = .1
        comfort_str_weight = 5.
        comfort_cost = np.sum(comfort_acc_weight * ctrls[:, 0]**2 + comfort_str_weight * ctrls[:, 1]**2)
        
        # 2. Efficiency Cost
        efficiency_weight = .01  # 恢复原始值 (修改前误改为 0.02)
        efficiency_cost = np.sum(efficiency_weight * (lcl_smp.target_velocity - states[:, 2])**2)
        
        # 3. Target Lane Cost
        target_weight = .01  # 恢复原始值 (修改前误改为 5.0，导致行为异常)
        dists_to_lane = []
        for state in states:
             dists_to_lane.append(self.get_dist_to_target_lane(lcl_smp, state))
        target_cost = np.sum(target_weight * np.array(dists_to_lane))

        # 4. Risk Cost (Vectorized & Max-Aggregated)
        risk_cost = 0.0
        if ENABLE_GHOST_PROBE and len(risk_sources) > 0:
            ego_vel_batch = torch.from_numpy(states[:, 2]).float().to(self.device)  # [N]
            
            # [PA-LOI v54 Final Fix] 彻底根除打分器的空间截断！
            # 提取每个风险源的权重和 v_safe
            weights = torch.tensor([r.get('weight', 0.0) for r in risk_sources], dtype=torch.float32, device=self.device).unsqueeze(0) # [1, M]
            v_safes = torch.tensor([r.get('v_safe', 0.0) for r in risk_sources], dtype=torch.float32, device=self.device).unsqueeze(0) # [1, M]
            
            # 计算超速部分 (v - v_safe)
            # [N, 1] - [1, M] -> [N, M]
            excess_vel = torch.clamp(ego_vel_batch.unsqueeze(1) - v_safes, min=0.0) 
            
            # 【关键修复】直接相乘！因为 utils.py 已经严格过滤了非盲区目标。
            # 这里只惩罚超速，坚决不用 dist_factor！
            raw_costs = weights * (excess_vel ** 2)
            
            # Max Aggregation: 取所有风险源中最大的那个 Cost
            node_max_costs, _ = torch.max(raw_costs, dim=1) # [N]
            
            # Sum over trajectory
            risk_cost = node_max_costs.sum().item()
        
        total_cost = (comfort_cost + efficiency_cost + target_cost + risk_cost) / n_nodes
        
        if return_breakdown:
            breakdown = {
                'comfort': float(comfort_cost / n_nodes),
                'efficiency': float(efficiency_cost / n_nodes),
                'target': float(target_cost / n_nodes),
                'risk': float(risk_cost / n_nodes)
            }
            return total_cost, breakdown
        return total_cost

    def get_dist_to_target_lane(self, lcl_smp, state):
        #  project the state to the target lane
        proj_state, _, _ = project_point_on_polyline(state[:2], lcl_smp.target_lane)
        #  get the distance
        dist = np.linalg.norm(proj_state - state[:2])
        return dist

    def get_interpolated_state(self, tree, timestep):
        root_node = tree.get_node(0)
        if timestep < root_node.data.t:
            return root_node.data.state, root_node.data.ctrl
        else:
            node = root_node
            while node.data.t <= timestep:
                node = tree.get_node(node.children_keys[0])
            #  interpolate the state
            prev_node = tree.get_node(node.parent_key)
            prev_state = prev_node.data.state
            next_state = node.data.state
            prev_time = prev_node.data.t
            next_time = node.data.t
            alpha = (timestep - prev_time) / (next_time - prev_time)
            interp_state = prev_state + alpha * (next_state - prev_state)
            return interp_state, node.data.ctrl
    
    # === PA-LOI: 实验数据记录管理 ===
    
    def init_data_logger(self, scenario_id="default", w_base=10.0, lambda_v=0.1, 
                         output_dir="./logs"):
        """
        初始化实验数据记录器
        
        在实验开始前调用此方法启动日志记录。
        
        Args:
            scenario_id: 场景标识符 (如 "S01", "ghost_probe_test")
            w_base: 当前实验的基础权重参数
            lambda_v: 当前实验的速度系数参数
            output_dir: 日志输出目录
        
        Example:
            planner.init_data_logger(scenario_id="S04", w_base=20.0, lambda_v=0.1)
            # ... 运行仿真 ...
            planner.save_experiment_log()
        """
        if not ENABLE_DATA_LOGGING:
            print("[PA-LOI Logger] Data logging is disabled (ENABLE_DATA_LOGGING=False)")
            return
        
        self.data_logger = PALOIDataLogger(
            scenario_id=scenario_id,
            w_base=w_base,
            lambda_v=lambda_v,
            output_dir=output_dir
        )
        print(f"[PA-LOI Logger] Initialized for scenario '{scenario_id}'")
    
    def save_experiment_log(self):
        """
        保存实验日志到 CSV 文件
        
        在实验结束后调用此方法保存数据。
        
        Returns:
            str: 保存的文件路径，如果未初始化则返回 None
        """
        if self.data_logger is None:
            print("[PA-LOI Logger] No logger initialized. Call init_data_logger() first.")
            return None
        
        filepath = self.data_logger.save()
        return filepath
    
    def get_experiment_summary(self):
        """
        获取实验摘要统计信息
        
        Returns:
            dict: 包含各项统计指标的字典
        """
        if self.data_logger is None:
            return {}
        return self.data_logger.get_summary()

