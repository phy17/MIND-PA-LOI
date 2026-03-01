#!/usr/bin/env python3
"""
Diagnostic script: visualize AV trajectory + occluder + ambush point geometry
for new ghost probe scenarios. This helps debug why LongDist is always negative.
"""
import sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import Simulator

# Test scene 0007df76
CONFIG = 'configs/ghost_experiment_vehicle.json'

sim = Simulator(CONFIG)
sim.init_sim()

# Find AV
av_agent = next((a for a in sim.agents if a.id == 'AV'), None)
ego_traj = av_agent.traj_info[0]  # [N, 2]
ego_heading = av_agent.traj_info[1]  # [N]

print(f"AV trajectory: {len(ego_traj)} points")
print(f"AV start: {ego_traj[0]}")
print(f"AV end: {ego_traj[-1]}")

# Calculate AV velocity at each point
for i in [0, 20, 40, 60, 80, 100]:
    if i < len(ego_traj) - 1:
        dx = np.linalg.norm(ego_traj[min(i+5, len(ego_traj)-1)] - ego_traj[i])
        dt = 5 * 0.1
        speed = dx / dt
        heading_deg = np.degrees(ego_heading[i])
        print(f"  idx={i}: pos={ego_traj[i]}, speed≈{speed:.1f}m/s, heading={heading_deg:.1f}°")

# Find static vehicles near trajectory (same logic as plan_ambush)
from av2.datasets.motion_forecasting.data_schema import ObjectType

print("\n--- Static vehicles near AV trajectory ---")
for agent in sim.agents:
    if agent.id == 'AV':
        continue
    if agent.type not in [ObjectType.VEHICLE, ObjectType.BUS]:
        continue
    agent_vel = agent.traj_info[2][0]
    if agent_vel > 0.5:
        continue
    
    agent_pos = agent.traj_info[0][0]
    distances = np.linalg.norm(ego_traj - agent_pos, axis=1)
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    
    if 1.5 < min_dist < 8.0:
        ego_at_pass = ego_traj[min_idx]
        ego_heading_at_pass = ego_heading[min_idx]
        drive_dir = np.array([np.cos(ego_heading_at_pass), np.sin(ego_heading_at_pass)])
        
        # Calculate lane direction from trajectory
        p_prev = ego_traj[max(0, min_idx-5)]
        p_next = ego_traj[min(len(ego_traj)-1, min_idx+5)]
        lane_dir = p_next - p_prev
        lane_dir = lane_dir / (np.linalg.norm(lane_dir) + 1e-6)
        
        # Vector from ego to occluder
        vec_to_occ = agent_pos - ego_at_pass
        
        # Where would target_pos be? (plan_ambush logic)
        # target is offset by front_edge_offset ALONG lane_dir from ego_pos_at_impact
        front_edge_offset = 2.0
        accumulated_dist = 0.0
        target_idx = min_idx
        for j in range(min_idx, len(ego_traj) - 1):
            accumulated_dist += np.linalg.norm(ego_traj[j+1] - ego_traj[j])
            if accumulated_dist >= front_edge_offset:
                target_idx = j + 1
                break
        target_pos = ego_traj[target_idx]
        
        # What's the longitudinal dist from AV START to target?
        # This tells us: when AV reaches enable_timestep=4.0s, has it already passed?
        dist_start_to_target = 0
        for j in range(0, target_idx):
            dist_start_to_target += np.linalg.norm(ego_traj[j+1] - ego_traj[j])
        
        # AV position at t=4.0s (enable_timestep)
        # At 10Hz, 4.0s = 40 frames
        av_pos_at_enable = ego_traj[min(40, len(ego_traj)-1)]
        
        # Distance from AV at enable to target
        vec_enable_to_target = target_pos - av_pos_at_enable
        heading_at_enable = ego_heading[min(40, len(ego_traj)-1)]
        drive_dir_at_enable = np.array([np.cos(heading_at_enable), np.sin(heading_at_enable)])
        long_dist_at_enable = np.dot(vec_enable_to_target, drive_dir_at_enable)
        
        print(f"\n  Agent {agent.id} ({agent.type})")
        print(f"    Position: {agent_pos}")
        print(f"    Dist to AV traj: {min_dist:.2f}m at traj idx {min_idx}")
        print(f"    Target pos (ambush target): {target_pos} at traj idx {target_idx}")
        print(f"    Distance from AV start to target: {dist_start_to_target:.1f}m")
        print(f"    AV pos at enable (t=4s, idx=40): {av_pos_at_enable}")
        print(f"    LongDist from AV@enable to target: {long_dist_at_enable:.2f}m")
        print(f"    → {'TARGET IS AHEAD ✅' if long_dist_at_enable > 0 else 'TARGET IS BEHIND ❌'}")
        
        # How far has AV traveled by t=4s?
        dist_by_enable = sum(np.linalg.norm(ego_traj[j+1] - ego_traj[j]) for j in range(min(40, len(ego_traj)-1)))
        print(f"    AV distance traveled by t=4s: {dist_by_enable:.1f}m")

print("\nDone!")
