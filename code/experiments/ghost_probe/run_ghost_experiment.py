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
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from simulator import Simulator
from agent import NonReactiveAgent, AgentColor
from common.bbox import PedestrianBBox
from av2.datasets.motion_forecasting.data_schema import ObjectType
import planners.mind.planner as planner_module


def _as_array(value, name):
    if value is None:
        raise ValueError(f"Missing ghost injection field: {name}")
    arr = np.array(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"Ghost injection field {name} must be a 2D point/vector, got {arr!r}")
    return arr


def ghost_config_from_spec(spec, pedestrian_speed_override=None, trigger_distance_override=None):
    """Convert a JSONL ghost-injection row into runtime simulator geometry."""
    seq_id = spec.get("seq_id") or spec.get("scenario_id")
    target_pos = _as_array(spec.get("target_pos", spec.get("p_cross")), "target_pos/p_cross")
    ambush_pos = _as_array(spec.get("ambush_pos", spec.get("ghost_start")), "ambush_pos/ghost_start")
    approach_dir = _as_array(spec.get("approach_dir", spec.get("ghost_direction")), "approach_dir/ghost_direction")
    approach_norm = np.linalg.norm(approach_dir)
    if approach_norm < 1e-9:
        raise ValueError("Ghost injection approach direction has near-zero length")
    approach_dir = approach_dir / approach_norm

    trigger_distance = trigger_distance_override
    if trigger_distance is None:
        trigger_distance = float(spec.get("trigger_distance_m", 4.5))

    pedestrian_speed = pedestrian_speed_override
    if pedestrian_speed is None:
        pedestrian_speed = float(spec.get("pedestrian_speed_ms", 2.0))

    return {
        "seq_id": seq_id,
        "ambush_pos": ambush_pos,
        "target_pos": target_pos,
        "path_idx": int(spec.get("cross_index", -1)),
        "occluder_id": spec.get("occluder_track_id"),
        "approach_dir": approach_dir,
        "trigger_point": spec.get("trigger_point"),
        "reference_trigger_frame": spec.get("reference_trigger_frame"),
        "reference_trigger_time_s": spec.get("reference_trigger_time_s"),
        "trigger_distance_m": trigger_distance,
        "pedestrian_speed_ms": pedestrian_speed,
        "ghost_spawn_mode": spec.get("ghost_spawn_mode", "dash"),
        "source_spec": spec,
    }


def _project_s_on_polyline(point, polyline, cum_s):
    """Project a 2D point onto a polyline and return arc-length coordinate."""
    point = np.array(point, dtype=float)
    best_s = float(cum_s[0])
    best_dist = float("inf")
    for idx in range(len(polyline) - 1):
        start = polyline[idx]
        end = polyline[idx + 1]
        seg = end - start
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-9:
            continue
        ratio = float(np.clip(np.dot(point - start, seg) / seg_len2, 0.0, 1.0))
        proj = start + ratio * seg
        dist = float(np.linalg.norm(point - proj))
        if dist < best_dist:
            best_dist = dist
            best_s = float(cum_s[idx] + ratio * np.linalg.norm(seg))
    return best_s


def load_ghost_injection_spec(jsonl_path, seq_id=None, index=None):
    """Load one ghost-injection row by AV2 scenario id or 1-based row/index."""
    path = Path(jsonl_path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if index is not None:
        for rec in records:
            if rec.get("index") == index:
                return rec
        if 1 <= index <= len(records):
            return records[index - 1]
        raise ValueError(f"No ghost injection record index {index} in {path}")

    if seq_id is not None:
        for rec in records:
            rec_seq = rec.get("seq_id") or rec.get("scenario_id")
            if rec_seq == seq_id:
                return rec
        raise ValueError(f"No ghost injection record for seq_id={seq_id} in {path}")

    if len(records) != 1:
        raise ValueError(f"{path} has {len(records)} records; provide seq_id or index")
    return records[0]


class GhostProbeSimulator(Simulator):
    """
    Extended simulator that injects a ghost probe agent at runtime.
    The ghost agent is NOT created until the trigger condition is met,
    ensuring the planner has no prior knowledge of its existence.
    """
    
    def __init__(
        self,
        config_path,
        enable_ghost_probe_defense=False,
        ghost_injection_spec=None,
        ghost_injection_jsonl=None,
        ghost_injection_index=None,
        pedestrian_speed=None,
        strict_trigger_dist=None,
        trigger_mode="distance",
        trigger_min_frame=None,
        trigger_max_frame=None,
        ghost_spawn_mode=None,
    ):
        super().__init__(config_path)
        
        # Ghost configuration (computed in init_sim, agent created in run_sim)
        self.ghost_config = None
        self.ghost_spawned = False
        self.ghost_agent = None
        
        # Control the ENABLE_GHOST_PROBE flag in planner module
        self.enable_ghost_probe_defense = enable_ghost_probe_defense
        
        # Optional fixed JSONL/spec geometry. When present, runtime injection uses
        # the same ghost location that was rendered for review.
        if ghost_injection_spec is None and ghost_injection_jsonl is not None:
            ghost_injection_spec = load_ghost_injection_spec(
                ghost_injection_jsonl,
                seq_id=None if ghost_injection_index is not None else self.seq_id,
                index=ghost_injection_index,
            )
        self.ghost_injection_spec = ghost_injection_spec
        self._pedestrian_speed_override = pedestrian_speed
        self._trigger_distance_override = strict_trigger_dist
        self.ghost_spawn_mode = ghost_spawn_mode or (
            ghost_injection_spec.get("ghost_spawn_mode") if ghost_injection_spec else "dash"
        )
        self.trigger_mode = trigger_mode
        self.trigger_min_frame = trigger_min_frame
        self.trigger_max_frame = trigger_max_frame
        self._reference_trigger_frame_raw = None
        self._reference_trigger_frame = None

        # Parameters
        self.trigger_distance = 15.0  # meters
        # [物理编排] 假人速度设为 2.0 m/s
        self.pedestrian_speed = 2.0
        if pedestrian_speed is not None:
            self.pedestrian_speed = float(pedestrian_speed)
        if strict_trigger_dist is not None:
            self.strict_trigger_dist = float(strict_trigger_dist)
        self.min_ego_speed = 0.05      
        self.time_lead = 0.1          # seconds (compensate for timing error)

    def _resolve_trigger_window(self):
        min_frame = self.trigger_min_frame
        max_frame = self.trigger_max_frame
        if min_frame is None:
            min_frame = 0
        if max_frame is None:
            max_frame = self.sim_horizon - 1
        min_frame = int(max(0, min_frame))
        max_frame = int(min(self.sim_horizon - 1, max_frame))
        if max_frame < min_frame:
            max_frame = min_frame
        return min_frame, max_frame

    def _compute_reference_trigger_frame(self):
        """Schedule trigger from the original AV trajectory, not closed-loop ego."""
        if not self.ghost_config:
            return None

        explicit_frame = self.ghost_config.get("reference_trigger_frame")
        if explicit_frame is not None:
            raw_frame = int(round(float(explicit_frame)))
        else:
            ego_agent = next((a for a in self.agents if a.id == "AV"), None)
            if ego_agent is None:
                return None
            ref_traj = np.array(ego_agent.traj_info[0], dtype=float)
            if len(ref_traj) < 2:
                return None
            seg_lens = np.linalg.norm(np.diff(ref_traj, axis=0), axis=1)
            cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
            target_s = _project_s_on_polyline(self.ghost_config["target_pos"], ref_traj, cum_s)
            trigger_s = max(float(cum_s[0]), target_s - float(self.strict_trigger_dist))
            raw_frame = int(np.searchsorted(cum_s, trigger_s, side="left"))

        min_frame, max_frame = self._resolve_trigger_window()
        frame = int(np.clip(raw_frame, min_frame, max_frame))
        self._reference_trigger_frame_raw = raw_frame
        self._reference_trigger_frame = frame
        self.ghost_config["reference_trigger_frame_raw"] = raw_frame
        self.ghost_config["reference_trigger_frame"] = frame
        self.ghost_config["reference_trigger_time_s"] = frame * self.sim_step
        self.ghost_config["trigger_window"] = [min_frame, max_frame]
        return frame
        
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
        
        # Load fixed geometry from JSONL when available; otherwise keep the
        # legacy online ambush planner for old demos.
        if self.ghost_injection_spec is not None:
            self.ghost_config = ghost_config_from_spec(
                self.ghost_injection_spec,
                pedestrian_speed_override=self._pedestrian_speed_override,
                trigger_distance_override=self._trigger_distance_override,
            )
            spec_seq = self.ghost_config.get("seq_id")
            if spec_seq is not None and spec_seq != self.seq_id:
                raise ValueError(f"Ghost spec seq_id={spec_seq} does not match config seq_id={self.seq_id}")
            self.pedestrian_speed = float(self.ghost_config["pedestrian_speed_ms"])
            self.strict_trigger_dist = float(self.ghost_config["trigger_distance_m"])
            print(
                "[GHOST_EXP] Loaded JSONL ghost geometry: "
                f"ambush={self.ghost_config['ambush_pos']}, "
                f"target={self.ghost_config['target_pos']}, "
                f"trigger={self.strict_trigger_dist}m, speed={self.pedestrian_speed}m/s, "
                f"spawn_mode={self.ghost_spawn_mode}"
            )
        else:
            self.ghost_config = self.plan_ambush()
        if self.ghost_config is None:
            print("[GHOST_EXP] WARNING: No valid ambush point found!")
        else:
            if self.trigger_mode in {"reference_time", "reference_frame", "scheduled"}:
                frame = self._compute_reference_trigger_frame()
                if frame is None:
                    print("[GHOST_EXP] WARNING: Could not compute reference trigger frame; falling back to distance trigger")
                    self.trigger_mode = "distance"
                else:
                    raw = self._reference_trigger_frame_raw
                    min_frame, max_frame = self._resolve_trigger_window()
                    print(
                        "[GHOST_EXP] Reference-scheduled trigger: "
                        f"raw_frame={raw}, clamped_frame={frame}, "
                        f"time={frame * self.sim_step:.2f}s, window=[{min_frame}, {max_frame}]"
                    )
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
            
            # Skip occluders that AV will pass before planning starts
            # enable_timestep=4.0s @ 50Hz = frame 200, add buffer for approach
            min_reachable_idx = int(getattr(self, '_min_traj_idx', 250))
            if min_idx < min_reachable_idx:
                continue
            
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
        front_edge_offset = vehicle_half_length - 0.5  # +2.0m
        
        # [精准修正] 严格沿着 ego_traj (即那条红线) 计算目标点，解决弯道目标点偏离的问题
        accumulated_dist = 0.0
        target_idx = best_path_idx
        for i in range(best_path_idx, len(ego_traj) - 1):
            accumulated_dist += np.linalg.norm(ego_traj[i+1] - ego_traj[i])
            if accumulated_dist >= front_edge_offset:
                target_idx = i + 1
                break
        target_pos = ego_traj[target_idx]
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
                                if agent.id == 'AV':
                                    print(f"  [GHOST_EXP] Recording planner failure as a collision for testing purposes.", flush=True)
                                    self.collision_log.append({
                                        "timestamp": float(self.sim_time),
                                        "frame_idx": step_idx,
                                        "ego_state": ego_agent.state.tolist() if ego_agent else [],
                                        "ego_vel": float(ego_agent.state[2]) if ego_agent else 0.0,
                                        "other_id": "PLANNER_CRASH",
                                        "other_type": "ERROR",
                                        "other_state": [],
                                        "collision_msg": "Planner crashed (NaNs/Failed optimization)."
                                    })
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

            # --- Ghost Tracking Log ---
            if self.ghost_spawned and self.ghost_agent and step_idx % 10 == 0:
                gh_pos = self.ghost_agent.state[:2]
                target_p = self.ghost_config['target_pos']
                dist_to_target = float(np.linalg.norm(gh_pos - target_p))
                gh_vel = float(self.ghost_agent.state[2])
                status = "STOPPED on Red Line" if gh_vel < 0.1 else "MOVING to Red Line"
                print(f"[GHOST TRACKING] t={self.sim_time:.2f}s | Status: {status} | Vel: {gh_vel:.2f}m/s | Dist to Red Line: {dist_to_target:.2f}m", flush=True)

            self.frames.append(frame)
            self.sim_time += self.sim_step

            if terminated:
                print("Simulation terminated!", flush=True)
                break
                
        print(f"[GHOST_EXP] Simulation complete. Ghost spawned: {self.ghost_spawned}, Collisions: {len(self.collision_log)}", flush=True)
        
        # Save Data Log
        av_agent = next((a for a in self.agents if a.id == 'AV'), None)
        log_path = None
        logger_collision_count = None
        logger_frame_count = None
        if av_agent and hasattr(av_agent, 'planner') and hasattr(av_agent.planner, 'save_experiment_log'):
            # Ground truth collision is from simulator. Sync it back to planner logger summary.
            if getattr(av_agent.planner, 'data_logger', None) is not None:
                av_agent.planner.data_logger.collision_count = len(self.collision_log)
                logger_collision_count = int(getattr(av_agent.planner.data_logger, 'collision_count', 0))
                logger_frame_count = int(getattr(av_agent.planner.data_logger, 'frame_count', 0))
            log_path = av_agent.planner.save_experiment_log()
            if log_path:
                print(f"[GHOST_EXP] Log saved to: {log_path}")

        # Canonical per-run summary (ground-truth first, with consistency fields)
        summary_path = Path(self.output_dir) / "run_summary.json"
        collision_count_gt = len(self.collision_log)
        first_collision = self.collision_log[0] if self.collision_log else None
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sim_name": self.sim_name,
            "seq_id": self.seq_id,
            "output_dir": self.output_dir,
            "ghost_spawned": bool(self.ghost_spawned),
            "ghost_spawn_time_s": getattr(self, "_ghost_spawn_time", None),
            "ghost_spawn_mode": self.ghost_spawn_mode,
            "trigger_mode": self.trigger_mode,
            "reference_trigger_frame_raw": self._reference_trigger_frame_raw,
            "reference_trigger_frame": self._reference_trigger_frame,
            "trigger_window": self.ghost_config.get("trigger_window") if self.ghost_config else None,
            "collision_count_ground_truth": collision_count_gt,
            "first_collision": first_collision,
            "collision_log": self.collision_log,
            "planner_log_path": log_path,
            "planner_logger_collision_count": logger_collision_count,
            "planner_logger_frame_count": logger_frame_count,
            "planner_logger_consistent_with_ground_truth": (
                None if logger_collision_count is None else logger_collision_count == collision_count_gt
            )
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[GHOST_EXP] Run summary saved: {summary_path}")

        if av_agent and hasattr(av_agent, 'planner'):
            # Reset logger for next run
            av_agent.planner.data_logger = None
    
    def should_spawn_ghost(self, debug=False):
        ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
        if ego_agent is None or not ego_agent.is_enable:
            return False, "No active ego"

        current_frame = int(round(self.sim_time / self.sim_step))
        if self.trigger_mode in {"reference_time", "reference_frame", "scheduled"}:
            trigger_frame = self._reference_trigger_frame
            if trigger_frame is None:
                trigger_frame = self._compute_reference_trigger_frame()
            if trigger_frame is not None:
                raw = self._reference_trigger_frame_raw
                msg = (
                    f"ReferenceFrame: {current_frame}/{trigger_frame} "
                    f"(raw={raw}, t={trigger_frame * self.sim_step:.2f}s)"
                )
                if current_frame >= trigger_frame and not self.ghost_spawned:
                    return True, f"TRIGGERED! {msg}"
                return False, msg
            
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
        strict_trigger_dist = getattr(self, 'strict_trigger_dist', 4.5)
        
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
        self._ghost_spawn_time = float(self.sim_time)
        self._ego_vel_at_spawn = float(ego_agent.state[2])
        
        # ADAS target: the fixed lane-center p_cross from the JSONL geometry.
        target_pos = self.ghost_config['target_pos']
        spawn_mode = self.ghost_spawn_mode or self.ghost_config.get("ghost_spawn_mode", "dash")

        if spawn_mode in {"instant_center", "instant", "teleport_center"}:
            # Extreme sudden-appearance case: once the ego reaches the strict
            # longitudinal trigger distance, the pedestrian is already standing
            # at the lane-center conflict point. No lateral dash motion is used.
            start_pos = np.array(target_pos, dtype=float)
            approach_dir = np.array(self.ghost_config.get("approach_dir", [1.0, 0.0]), dtype=float)
            norm = np.linalg.norm(approach_dir)
            if norm < 1e-9:
                approach_dir = np.array([1.0, 0.0], dtype=float)
            else:
                approach_dir = approach_dir / norm
            print("[GHOST_EXP] Instant-center mode: pedestrian appears directly at lane center")
        else:
            # Default ghost-probe case: pedestrian starts at the occluder-side
            # ambush point and walks to the lane center.
            start_pos = self.ghost_config['ambush_pos']
            approach_dir = self.ghost_config['approach_dir']
        
        # Generate trajectory frames (remaining simulation time)
        remaining_frames = self.sim_horizon - int(self.sim_time / self.sim_step)
        traj_pos = []
        traj_ang = []
        traj_vel = []
        has_flag = []
        
        # Calculate heading (towards target, or along approach direction for an
        # instant standing pedestrian where start == target).
        direction = target_pos - start_pos
        if np.linalg.norm(direction) < 1e-9:
            direction = approach_dir
        heading = np.arctan2(direction[1], direction[0])
        
        stop_distance = np.linalg.norm(target_pos - start_pos)
        if spawn_mode in {"instant_center", "instant", "teleport_center"}:
            stop_time = 0.0
            print("[GHOST_EXP] Pedestrian: instant appearance at lane center, then STOPS")
        else:
            # 行人从 start_pos 移动到 target_pos（车道中心）后停下
            stop_time = stop_distance / self.pedestrian_speed  # 到达车道中心所需时间
            print(f"[GHOST_EXP] Pedestrian: {stop_distance:.2f}m to lane center, "
                  f"arrives in {stop_time:.2f}s, then STOPS")
        
        for i in range(remaining_frames + 50):  # Extra buffer
            t = i * self.sim_step
            if spawn_mode in {"instant_center", "instant", "teleport_center"}:
                pos = start_pos
                vel = 0.0
            elif t <= stop_time:
                # 还没到车道中心：匀速行走
                pos = start_pos + approach_dir * self.pedestrian_speed * t
                vel = self.pedestrian_speed
            else:
                # 到达车道中心：停住不动（愣在原地）
                pos = start_pos + approach_dir * stop_distance
                vel = 0.0
            traj_pos.append(pos)
            traj_ang.append(heading)
            traj_vel.append(vel)
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
                                float(traj_vel[0]), heading])
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
    sim_baseline.sim_horizon = 550  # 550 frames = 11s
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
    sim_improved.sim_horizon = 550 # 550 frames = 11s
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
