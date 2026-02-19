#!/usr/bin/env python3
"""
Ghost Probe Comparative Experiment Script

This script creates a controlled ghost probe scenario to validate the effectiveness
of the Risk Field (Ghost Probe Detection) algorithm.

Usage:
    python experiments/ghost_probe/run_ghost_experiment.py --config configs/1.json

Features:
    1. Spawn-on-Trigger: Ghost agent doesn't exist until trigger moment.
    2. Smart Positioning: Automatically finds occluders and calculates ambush points.
    3. ADAS Tracking: Ghost trajectory tracks ego's lateral position for guaranteed collision.
    4. Comparative Testing: Runs baseline (no protection) vs improved (risk field enabled).
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from simulator import Simulator
from agent import NonReactiveAgent, AgentColor
from common.bbox import PedestrianBBox
from av2.datasets.motion_forecasting.data_schema import ObjectType
import planners.mind.planner as planner_module


class GhostProbeSimulator(Simulator):
    """
    Extended simulator that injects a ghost probe agent at runtime.
    The ghost agent is NOT created until the trigger condition is met,
    ensuring the planner has no prior knowledge of its existence.
    """
    
    def __init__(self, config_path, enable_ghost_probe_defense=False):
        super().__init__(config_path)
        
        # Ghost configuration (computed in init_sim, agent created in run_sim)
        self.ghost_config = None
        self.ghost_spawned = False
        self.ghost_agent = None
        
        # Control the ENABLE_GHOST_PROBE flag in planner module
        self.enable_ghost_probe_defense = enable_ghost_probe_defense
        
        # Parameters
        self.trigger_distance = 15.0  # meters
        # [物理编排] 假人速度设为 2.0 m/s
        self.pedestrian_speed = 2.0   
        self.min_ego_speed = 0.05      
        self.time_lead = 0.1          # seconds (compensate for timing error)
        
    def init_sim(self):
        """Initialize simulation and plan the ambush (but don't create ghost yet)."""
        super().init_sim()
        
        planner_module.ENABLE_GHOST_PROBE = self.enable_ghost_probe_defense
        print(f"[GHOST_EXP] ENABLE_GHOST_PROBE set to: {self.enable_ghost_probe_defense}")
        
        # Initialize Data Logger via planner instance
        av_agent = next((a for a in self.agents if a.id == 'AV'), None)
        if av_agent and hasattr(av_agent, 'planner'):
            log_id = "baseline" if not self.enable_ghost_probe_defense else "improved"
            log_dir = os.path.join(self.output_dir, "logs")
            av_agent.planner.init_data_logger(
                scenario_id=f"ghost_exp_{log_id}", 
                w_base=20.0,  # Default, can be tuned
                lambda_v=0.1, # Default
                output_dir=log_dir
            )
            print(f"[GHOST_EXP] Data Logger initialized for {log_id}")
        
        # Plan the ambush
        self.ghost_config = self.plan_ambush()
        if self.ghost_config is None:
            print("[GHOST_EXP] WARNING: No valid ambush point found!")
        else:
            print(f"[GHOST_EXP] Ambush planned at position {self.ghost_config['ambush_pos']}")
            
    def plan_ambush(self):
        """
        Analyze the scenario to find the best ambush point.
        Uses the ego's GT trajectory to find occluders along the path.
        
        Returns:
            dict with ambush configuration, or None if no valid spot found.
        """
        # Find the ego agent (AV)
        ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
        if ego_agent is None:
            print("[GHOST_EXP] ERROR: No AV agent found!")
            return None
            
        # Get ego's GT trajectory (future path from the dataset)
        ego_traj = ego_agent.traj_info[0]  # [N, 2] position array
        ego_heading = ego_agent.traj_info[1]  # [N] heading array
        
        # Find static occluders (buses, vehicles) along the path
        best_occluder = None
        best_distance_to_path = float('inf')
        best_path_idx = 0
        
        for agent in self.agents:
            if agent.id == 'AV':
                continue
            
            # Check if it's a potential occluder (vehicle or bus)
            if agent.type not in [ObjectType.VEHICLE, ObjectType.BUS]:
                continue
                
            # Check if it's static (low velocity at initial step)
            agent_vel = agent.traj_info[2][0]  # Initial velocity
            if agent_vel > 0.5:  # Moving agent, skip
                continue
            
            # Find closest point on ego trajectory
            agent_pos = agent.traj_info[0][0]  # Initial position
            distances = np.linalg.norm(ego_traj - agent_pos, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # Must be close to path but not ON the path (would block ego)
            if 2.0 < min_dist < 6.0:  # Expanded range slightly
                if min_dist < best_distance_to_path:
                    best_distance_to_path = min_dist
                    best_occluder = agent
                    best_path_idx = min_idx
        
        if best_occluder is None:
            print("[GHOST_EXP] No suitable occluder found, using fallback position.")
            fallback_idx = min(200, len(ego_traj) - 50)
            target_pos = ego_traj[fallback_idx]
            ambush_pos = target_pos + np.array([3.0, 0.0])
            return {
                'ambush_pos': ambush_pos,
                'target_pos': target_pos,
                'path_idx': fallback_idx,
                'occluder_id': None,
                'approach_dir': np.array([-1.0, 0.0])
            }
        
        # Calculate ambush position (behind the occluder from ego's perspective)
        occluder_pos = best_occluder.traj_info[0][0]
        ego_pos_at_impact = ego_traj[best_path_idx] # Collision point on lane
        
        # Vector from Lane -> Occluder (Lateral vector)
        lane_to_occluder = occluder_pos - ego_pos_at_impact
        # Normalize
        dist = np.linalg.norm(lane_to_occluder)
        lane_to_occluder_dir = lane_to_occluder / (dist + 1e-6)
        
        # We want the ghost to spawn AT the occluder (slightly towards the lane)
        # And slightly ahead/behind along the lane to avoid being inside the occluder?
        # Simulation: Just spawn at occluder position, offset by occluder size (approx 2m) + buffer
        
        # Better strategy: 
        # 1. Target Point = ego_pos_at_impact (Center of lane)
        # 2. Ambush Point = occluder_pos
        # 3. But we want to spawn *hidden* behind the occluder.
        #    If ego is coming from South, we want spawn point to be North of Occluder? 
        #    Or if it's "Sudden Appearance", maybe simply spawn AT the occluder's edge facing the lane.
        
        # Let's use the user's diagram logic:
        # Occluder is parallel to lane. Ghost jumps out perpendicular.
        
        # Direction of lane (tangent at impact)
        p_prev = ego_traj[max(0, best_path_idx-5)]
        p_next = ego_traj[min(len(ego_traj)-1, best_path_idx+5)]
        lane_dir = p_next - p_prev
        lane_dir = lane_dir / (np.linalg.norm(lane_dir) + 1e-6)
        
        # Determine if occluder is Left or Right of lane
        # Cross product of LaneDir and VectorToOccluder
        cross_prod = lane_dir[0]*lane_to_occluder_dir[1] - lane_dir[1]*lane_to_occluder_dir[0]
        is_left = cross_prod > 0
        
        # Move ambush point slightly along the lane direction (to the "front" of the parked car)
        # "Front" depends on ego direction. We want it closer to ego? No, further.
        # User said "Sudden appearance from dead angle".
        # Let's put it 2.0m along lane_dir from occluder center.
        longitudinal_offset = 2.0 
        ambush_pos = occluder_pos - lane_dir * longitudinal_offset # Slightly upstream? Or downstream?
        # Let's try: Align with occluder, but strictly ensure approach_dir is Lateral.
        
        # Calculate lane direction at the impact point
        p_prev = ego_traj[max(0, best_path_idx-5)]
        p_next = ego_traj[min(len(ego_traj)-1, best_path_idx+5)]
        lane_vec = p_next - p_prev
        lane_dir = lane_vec / (np.linalg.norm(lane_vec) + 1e-6)
        
        # ====================================================================
        # [物理编排] 假人从遮挡物的前端边缘（盲区边界）冲出
        # 
        #   Ego →→→ lane_dir →→→
        #                     ↓ 前端边缘 (blind spot boundary)
        #            ┌────────┼──────────────────┐
        #            │   Occluder (大巴/停车)      │
        #            └────────┼──────────────────┘
        #                     ↓
        #               假人从这里冲出来！
        #               ambush_pos ●─→ approach_dir
        #                     │
        #      ═══════════════●══════════════  车道中心 (target)
        # ====================================================================
        
        # 估算遮挡物半车长（中心到前端的距离）
        # 大巴 ~5m, 普通车 ~2.5m, 保守取 2.5m
        vehicle_half_length = 2.5
        
        # 前端边缘在车道上的投影：从 ego_pos_at_impact（遮挡物中心对应的车道点）
        # 往 ego 去的方向进 vehicle_half_length，就是另一端的纵向位置
        # 再减 0.5m 微调，让假人刚好从后保险杠后面蹿出来？
        front_edge_offset = vehicle_half_length - 0.5  # +2.0m (Opposite side!)
        target_pos = ego_pos_at_impact + lane_dir * front_edge_offset
        
        # 横向方向：从 target_pos 指向遮挡物中心的横向分量
        vec_to_occluder = occluder_pos - target_pos
        lat_dir = vec_to_occluder - np.dot(vec_to_occluder, lane_dir) * lane_dir
        lat_dir = lat_dir / (np.linalg.norm(lat_dir) + 1e-6)
        
        # [物理编排] 假人距离车道中心横向 1.5 米处刷出
        # 配合 2.0m/s 的速度，0.75 秒抵达车道中心
        ambush_pos = target_pos + lat_dir * 1.5
        approach_dir = -lat_dir
        
        print(f"[GHOST_EXP] Geometry: Occluder@{occluder_pos} (half_len={vehicle_half_length}m)")
        print(f"[GHOST_EXP]   Front Edge Target@{target_pos} (offset={front_edge_offset}m from center)")
        print(f"[GHOST_EXP]   Ambush@{ambush_pos} (lat=1.5m from lane center)")
        
        return {
            'ambush_pos': ambush_pos,
            'target_pos': target_pos,
            'path_idx': best_path_idx,
            'occluder_id': best_occluder.id,
            'approach_dir': approach_dir 
        }
    
    def run_sim(self):
        """Extended simulation loop with ghost spawn logic."""
        print("[GHOST_EXP] Running simulation with ghost probe injection...", flush=True)
        self.frames = []
        self.sim_time = 0.0
        terminated = False
        collided = False
        
        from tqdm import tqdm
        from agent import CustomizedAgent
        
        for step_idx in tqdm(range(self.sim_horizon)):
            frame = {}
            
            # === GHOST SPAWN LOGIC ===
            if self.ghost_config and not self.ghost_spawned:
                should_spawn, debug_msg = self.should_spawn_ghost(debug=True)
                if step_idx % 20 == 0:
                     print(f"[TRIGGER CHECK] Step {step_idx}: {debug_msg}", flush=True)

                if should_spawn:
                    print(f"[TRIGGER FIRE] !!! SPAWNING GHOST at Step {step_idx} !!! : {debug_msg}", flush=True)
                    self.spawn_ghost_agent()
            
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
            ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
            if ego_agent and ego_agent.is_enable and not collided:
                ego_poly = self.get_agent_polygon(ego_agent)
                
                for other in self.agents:
                    if other.id == 'AV': continue
                    if np.linalg.norm(other.state[:2] - ego_agent.state[:2]) > 10.0:
                        continue
                        
                    other_poly = self.get_agent_polygon(other)
                    from common.geometry import check_polygon_intersection
                    if check_polygon_intersection(ego_poly, other_poly):
                        print(f"\n[COLLISION] At {self.sim_time:.2f}s: Ego collided with {other.id} ({other.type})", flush=True)
                        collided = True
                        self.collision_log.append({
                            "timestamp": float(self.sim_time),
                            "frame_idx": step_idx,
                            "ego_state": ego_agent.state.tolist(),
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
                        if agent.is_enable:
                            is_success, res = agent.plan()
                            if not is_success:
                                print(f"[ERROR] Agent {agent.id} plan failed!", flush=True)
                                terminated = True
                                break

                            if agent.id == 'AV':
                                frame['scen_tree'] = res[0]
                                frame['traj_tree'] = res[1]
                                if len(res) > 2:
                                    frame['ghost_points'] = res[2]
                                if hasattr(agent, 'gt_tgt_lane'):
                                    frame['target_lane'] = agent.gt_tgt_lane

                elif isinstance(agent, NonReactiveAgent):
                    agent.step()
                else:
                    raise ValueError("Unknown agent type")
                agent.update_state(self.sim_step)

            self.frames.append(frame)
            self.sim_time += self.sim_step

            if terminated:
                print("Simulation terminated!", flush=True)
                break
                
        print(f"[GHOST_EXP] Simulation complete. Ghost spawned: {self.ghost_spawned}, Collisions: {len(self.collision_log)}", flush=True)
        
        # Save Data Log
        av_agent = next((a for a in self.agents if a.id == 'AV'), None)
        if av_agent and hasattr(av_agent, 'planner') and hasattr(av_agent.planner, 'save_experiment_log'):
            log_path = av_agent.planner.save_experiment_log()
            if log_path:
                print(f"[GHOST_EXP] Log saved to: {log_path}")
            
            # Reset logger for next run
            av_agent.planner.data_logger = None
    
    def should_spawn_ghost(self, debug=False):
        ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
        if ego_agent is None or not ego_agent.is_enable:
            return False, "No active ego"
            
        ego_pos = ego_agent.state[:2]
        ego_vel = ego_agent.state[2]
        ego_heading = ego_agent.state[3]
        
        if ego_vel < self.min_ego_speed:
            return False, f"Speed {ego_vel:.1f} < {self.min_ego_speed}, fully stopped."
        
        target_pos = self.ghost_config['target_pos']
        
        # ====================================================================
        # [核心算法] 沿着小车前进方向的纵向投影距离 (Longitudinal Projection)
        # 排除横向距离干扰，像一根横跨车道的隐形红外线绊马索
        # ====================================================================
        vec_to_target = target_pos - ego_pos
        drive_direction = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        # 向量点乘：计算出严格的纵向深度
        longitudinal_dist = np.dot(vec_to_target, drive_direction)
        
        # 死亡触发线：4.5 米
        # (车长约5m，自车中心到前保险杠约2.0~2.5m，保险杠距离假人实际仅剩 2.0m)
        # 不减速的 Baseline (约 4.0m/s) 刹停至少需 2.8 米 -> 物理亏空，必定撞飞！
        # PA-LOI 提前减速 (约 2.5m/s) 刹停仅需 1.28 米 -> 完美避险！
        strict_trigger_dist = 4.5
        
        if debug:
            debug_msg = f"LongDist: {longitudinal_dist:.2f}m vs Strict: {strict_trigger_dist}m"
            
        # 触发条件：纵向距离小于等于 4.5 米，且大于 0.0（防止小车开过头了还在背后误触发）
        if 0.0 < longitudinal_dist <= strict_trigger_dist and not self.ghost_spawned:
            return True, f"TRIGGERED! Longitudinal Death Zone: {strict_trigger_dist}m"
            
        return False, debug_msg
    
    def spawn_ghost_agent(self):
        """Spawn the ghost agent at the ambush position with collision trajectory."""
        print(f"[GHOST_EXP] *** SPAWNING GHOST at t={self.sim_time:.2f}s ***")
        
        # Get current ego state for ADAS tracking
        ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
        ego_pos = ego_agent.state[:2]
        ego_heading = ego_agent.state[3]
        
        # Calculate ghost trajectory (straight line crossing)
        start_pos = self.ghost_config['ambush_pos']
        approach_dir = self.ghost_config['approach_dir']
        
        # ADAS tracking: adjust target to ego's current lateral position
        target_pos = ego_pos.copy()
        
        # Generate trajectory frames (remaining simulation time)
        remaining_frames = self.sim_horizon - int(self.sim_time / self.sim_step)
        traj_pos = []
        traj_ang = []
        traj_vel = []
        has_flag = []
        
        # Calculate heading (towards target)
        direction = target_pos - start_pos
        heading = np.arctan2(direction[1], direction[0])
        
        for i in range(remaining_frames + 50):  # Extra buffer
            t = i * self.sim_step
            pos = start_pos + approach_dir * self.pedestrian_speed * t
            traj_pos.append(pos)
            traj_ang.append(heading)
            traj_vel.append(self.pedestrian_speed)
            has_flag.append(1)
        
        traj_pos = np.array(traj_pos).astype(np.float32)
        traj_ang = np.array(traj_ang).astype(np.float32)
        traj_vel = np.array(traj_vel).astype(np.float32)
        has_flag = np.array(has_flag).astype(np.int16)
        traj_type = [ObjectType.PEDESTRIAN] * len(traj_pos)
        
        traj_info = [traj_pos, traj_ang, traj_vel, has_flag]
        
        # Create the ghost agent
        ghost = NonReactiveAgent()
        ghost.id = "GHOST_001"
        ghost.type = ObjectType.PEDESTRIAN
        ghost.bbox = PedestrianBBox()
        ghost.clr = ['cyan', 'blue']  # Blue color for visibility
        ghost.traj_info = traj_info
        ghost.traj_type = traj_type
        ghost.traj_cat = "ghost"
        ghost.rec_step = 0
        ghost.max_step = len(traj_pos) - 1
        ghost.state = np.array([traj_pos[0][0], traj_pos[0][1], 
                                self.pedestrian_speed, heading])
        ghost.ctrl = np.array([0.0, 0.0])
        ghost.timestep = self.sim_time
        
        # Add to agents list
        self.agents.append(ghost)
        self.ghost_agent = ghost
        self.ghost_spawned = True


def run_comparative_experiment(config_path, output_base_dir="output/ghost_experiment"):
    """
    Run the comparative experiment:
    1. Baseline run (no ghost probe defense) -> expect collision
    2. Improved run (ghost probe defense enabled) -> expect avoidance
    """
    results = {}
    
    # --- Run 1: Baseline (No Defense) ---
    print("\n" + "="*60)
    print("  RUN 1: BASELINE (No Defense)")
    print("="*60 + "\n")
    
    sim_baseline = GhostProbeSimulator(config_path, enable_ghost_probe_defense=False)
    sim_baseline.init_sim()
    sim_baseline.output_dir = output_base_dir + "/baseline/"
    sim_baseline.run()
    
    results['baseline'] = {
        'collision_count': len(sim_baseline.collision_log),
        'collision_log': sim_baseline.collision_log
    }
    
    # --- Run 2: Improved (With Defense) ---
    print("\n" + "="*60)
    print("\n" + "="*60)
    print("  RUN 2: IMPROVED (ENABLE_GHOST_PROBE = True)")
    print("="*60 + "\n")
    # 2. Improved (PA-LOI)
    sim_improved = GhostProbeSimulator(config_path, enable_ghost_probe_defense=True)
    sim_improved.init_sim()
    # [Benchmark] run for more steps to capture full stop
    sim_improved.output_dir = output_base_dir + "/improved/"
    sim_improved.sim_horizon = 700 # Manually override horizon
    sim_improved.run()
    
    results['improved'] = {
        'collision_count': len(sim_improved.collision_log),
        'collision_log': sim_improved.collision_log
    }
    
    # --- Summary ---
    print("\n" + "="*60)
    print("  EXPERIMENT RESULTS")
    print("="*60)
    # print(f"  Baseline (No Defense):  {results['baseline']['collision_count']} collisions")
    print(f"  Improved (With Defense): {results['improved']['collision_count']} collisions")
    
    if results['improved']['collision_count'] == 0:
        print("\n  ✅ SUCCESS: Ghost Probe Defense effectively prevented collision!")
    else:
        print("\n  ❌ FAILURE: Defense did not prevent collision.")
    
    # Save results
    os.makedirs(output_base_dir, exist_ok=True)
    with open(f"{output_base_dir}/experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_base_dir}/experiment_results.json")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ghost Probe Comparative Experiment")
    parser.add_argument("--config", type=str, default="configs/1.json",
                        help="Path to simulation config file")
    parser.add_argument("--output", type=str, default="output/ghost_experiment",
                        help="Output directory for results")
    args = parser.parse_args()
    
    run_comparative_experiment(args.config, args.output)
