#!/usr/bin/env python3
"""
诊断: 为什么 baseline 在 230 帧附近必定崩溃？
单跑一次 baseline (无 PA-LOI, 有 AEB)，打印每一帧的详细状态
"""
import sys, os
import importlib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import planners.mind.planner as pm
importlib.reload(pm)

pm.ENABLE_GHOST_PROBE = False
pm.ENABLE_AEB = True
pm.ENABLE_DATA_LOGGING = False

from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator

sim = GhostProbeSimulator('configs/ghost_experiment_vehicle.json', 
                           enable_ghost_probe_defense=False)
sim.init_sim()
sim.strict_trigger_dist = 5.0
sim.output_dir = "output/debug_baseline_crash"
sim.sim_horizon = 300  # 只需要跑到崩溃点之后
sim.render = False

os.makedirs(sim.output_dir, exist_ok=True)

# 手动跑仿真循环，在每一帧打印详细信息
from agent import CustomizedAgent
from agent import CustomizedAgent, NonReactiveAgent
from tqdm import tqdm

print("Starting diagnostic simulation...")
sim.frames = []
sim.sim_time = 0.0

for step_idx in range(sim.sim_horizon):
    frame = {}
    
    # Ghost spawn logic (same as run_sim)
    if sim.ghost_config and not sim.ghost_spawned:
        should_spawn, debug_msg = sim.should_spawn_ghost(debug=True)
        if should_spawn:
            print(f"[TRIGGER FIRE] Step {step_idx}: {debug_msg}")
            sim.spawn_ghost_agent()
    
    # 获取 ego 状态
    ego_agent = next((a for a in sim.agents if a.id == 'AV'), None)
    
    # 在关键帧区间打印详细信息
    if ego_agent and ego_agent.is_enable and step_idx >= 195:
        ego_pos = ego_agent.state[:2]
        ego_vel = ego_agent.state[2]
        ego_heading = ego_agent.state[3]
        print(f"\n[DIAG] Step {step_idx} | t={sim.sim_time:.3f}s | "
              f"pos=({ego_pos[0]:.2f}, {ego_pos[1]:.2f}) | "
              f"vel={ego_vel:.3f} m/s | heading={ego_heading:.4f}")
    
    # Update agent observations
    agent_obs = []
    for agent in sim.agents:
        if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
            agent_obs.append(agent.observe())

    agent_gt = []
    for agent in sim.agents:
        if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
            agent_gt.append(agent.observe_no_noise())

    frame['agents'] = agent_gt
    
    # Planning
    terminated = False
    for agent in sim.agents:
        if isinstance(agent, CustomizedAgent):
            agent.check_enable(sim.sim_time)
            rec_tri, pl_tri = agent.check_trigger(sim.sim_time)
            if rec_tri:
                agent.step()
            if pl_tri:
                agent.update_observation(agent_obs)
                if agent.is_enable:
                    try:
                        is_success, res = agent.plan()
                        if not is_success:
                            print(f"\n[CRASH] Step {step_idx}: Planner returned failure!")
                            terminated = True
                            break
                    except Exception as e:
                        print(f"\n[CRASH] Step {step_idx}: Exception during planning: {e}")
                        import traceback
                        traceback.print_exc()
                        terminated = True
                        break
                    
                    if agent.id == 'AV':
                        frame['scen_tree'] = res[0]
                        frame['traj_tree'] = res[1]

        elif isinstance(agent, NonReactiveAgent):
            agent.step()
        agent.update_state(sim.sim_step)

    sim.frames.append(frame)
    sim.sim_time += sim.sim_step
    
    if terminated:
        print(f"\n[DIAG] Simulation terminated at step {step_idx}")
        break

print(f"\n[DIAG] Simulation ended at step {step_idx}, t={sim.sim_time:.2f}s")
print(f"[DIAG] Ghost spawned: {sim.ghost_spawned}")
print(f"[DIAG] Collisions: {len(sim.collision_log)}")
