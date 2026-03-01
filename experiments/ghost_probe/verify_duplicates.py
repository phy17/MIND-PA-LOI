#!/usr/bin/env python3
"""
验证重复数据点 - 碰撞即停版 (Early Exit on Collision)
"""

import sys, os, json, time, importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

VEHICLE_HALF_LENGTH = 2.25
OUTPUT_BASE = "output/verify_duplicates"


def run_single(trigger_dist, config_path, exp_name, enable_defense, enable_aeb):
    """跑单次实验，碰撞后立即停止"""
    import planners.mind.planner as pm
    importlib.reload(pm)
    pm.ENABLE_GHOST_PROBE = enable_defense
    pm.ENABLE_AEB = enable_aeb
    pm.ENABLE_DATA_LOGGING = False

    from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator

    class EarlyExitSimulator(GhostProbeSimulator):
        """碰撞后立即终止的仿真器"""
        def run_sim(self):
            self.frames = []
            self.sim_time = 0.0
            from agent import CustomizedAgent
            from tqdm import tqdm

            for step_idx in tqdm(range(self.sim_horizon), desc=exp_name):
                frame = {}

                # Ghost spawn logic
                if self.ghost_config and not self.ghost_spawned:
                    should_spawn, _ = self.should_spawn_ghost(debug=True)
                    if should_spawn:
                        print(f"\n[SPAWN] Ghost at step {step_idx}, t={self.sim_time:.2f}s")
                        self.spawn_ghost_agent()

                # Agent observations
                from agent import NonReactiveAgent
                agent_obs = []
                for agent in self.agents:
                    if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                        agent_obs.append(agent.observe())

                agent_gt = []
                for agent in self.agents:
                    if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                        agent_gt.append(agent.observe_no_noise())
                frame['agents'] = agent_gt

                # Collision check
                ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
                if ego_agent and ego_agent.is_enable:
                    ego_poly = self.get_agent_polygon(ego_agent)
                    for other in self.agents:
                        if other.id == 'AV': continue
                        import numpy as np
                        if np.linalg.norm(other.state[:2] - ego_agent.state[:2]) > 10.0:
                            continue
                        other_poly = self.get_agent_polygon(other)
                        from common.geometry import check_polygon_intersection
                        if check_polygon_intersection(ego_poly, other_poly):
                            vel = float(ego_agent.state[2])
                            self.collision_log.append({
                                "timestamp": float(self.sim_time),
                                "frame_idx": step_idx,
                                "ego_state": ego_agent.state.tolist(),
                                "ego_vel": vel,
                                "other_id": str(other.id),
                                "other_type": str(other.type),
                                "other_state": other.state.tolist(),
                            })
                            print(f"\n💥 COLLISION at t={self.sim_time:.2f}s, vel={vel:.6f} m/s")
                            print(f"   Early exit! No need to continue.")
                            return  # ← 碰撞即停！

                # Plan
                terminated = False
                for agent in self.agents:
                    if isinstance(agent, CustomizedAgent):
                        agent.check_enable(self.sim_time)
                        rec_tri, pl_tri = agent.check_trigger(self.sim_time)
                        if rec_tri: agent.step()
                        if pl_tri:
                            agent.update_observation(agent_obs)
                            if agent.is_enable:
                                is_success, res = agent.plan()
                                if not is_success:
                                    terminated = True
                                    break
                                if agent.id == 'AV':
                                    frame['scen_tree'] = res[0]
                                    frame['traj_tree'] = res[1]
                    elif isinstance(agent, NonReactiveAgent):
                        agent.step()
                    agent.update_state(self.sim_step)

                self.frames.append(frame)
                self.sim_time += self.sim_step
                if terminated: break

    sim = EarlyExitSimulator(config_path, enable_ghost_probe_defense=enable_defense)
    sim.init_sim()
    sim.strict_trigger_dist = trigger_dist
    sim.output_dir = os.path.join(OUTPUT_BASE, exp_name)
    sim.sim_horizon = 550
    sim.render = False
    os.makedirs(sim.output_dir, exist_ok=True)

    start = time.time()
    sim.run()
    elapsed = time.time() - start

    collisions = len(sim.collision_log)
    col_vel = sim.collision_log[0]['ego_vel'] if sim.collision_log else 0.0

    return {
        'exp_name': exp_name,
        'trigger_dist': trigger_dist,
        'bumper_dist': round(trigger_dist - VEHICLE_HALF_LENGTH, 2),
        'collisions': collisions,
        'vel_raw': col_vel,
        'vel_rounded': round(col_vel, 2),
        'elapsed_s': round(elapsed, 1),
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # d=4.3 已跑完: vel=2.878899569280918 → 2.88
    experiments = [
        (4.5, 'configs/ghost_experiment.json',         'intersection_aeb_d4.5', False, True),
        (3.5, 'configs/ghost_experiment_vehicle.json', 'straight_aeb_d3.5',     False, True),
        (4.0, 'configs/ghost_experiment_vehicle.json', 'straight_aeb_d4.0',     False, True),
    ]

    results = []
    for i, (dist, config, name, defense, aeb) in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/4] {name}")
        print(f"{'='*60}")
        r = run_single(dist, config, name, defense, aeb)
        results.append(r)
        print(f"  ✅ Done in {r['elapsed_s']}s → vel={r['vel_raw']:.10f}")

    # 汇总
    print(f"\n\n{'='*80}")
    print("  验证结果汇总")
    print(f"{'='*80}")
    print(f"{'实验':<28} {'碰撞':<6} {'速度(完整浮点)':<28} {'四舍五入':<10}")
    print("-" * 72)
    for r in results:
        col = "YES" if r['collisions'] > 0 else "NO"
        print(f"{r['exp_name']:<28} {col:<6} {r['vel_raw']:<28.15f} {r['vel_rounded']}")

    print(f"\n  路口 d=4.3 vs d=4.5 差异: {abs(results[0]['vel_raw'] - results[1]['vel_raw']):.15f}")
    print(f"  直道 d=3.5 vs d=4.0 差异: {abs(results[2]['vel_raw'] - results[3]['vel_raw']):.15f}")

    with open(os.path.join(OUTPUT_BASE, "verify_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {OUTPUT_BASE}/verify_results.json")


if __name__ == "__main__":
    main()
