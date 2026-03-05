#!/usr/bin/env python3
"""
No-Ghost Efficiency Test: Intersection Scenario
================================================
Purpose: Measure traffic efficiency impact of PA-LOI + AEB
         when NO pedestrian ever appears (no ghost probe).

Records per-frame: ego velocity
Computes: avg speed, min speed, slow frame % vs Vanilla MIND

Scene: ghost_experiment.json (intersection scenario)
Frames: 500 @ 10 Hz = 50 seconds
"""

import os
import sys
import csv
import time
import importlib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_DIR = 'output/no_ghost_efficiency_test'
CONFIG     = 'configs/ghost_experiment.json'
N_FRAMES   = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_one_trial(label, enable_pa_loi, enable_aeb):
    """Run a single 500-frame trial and return per-frame data."""
    import planners.mind.planner as pm
    importlib.reload(pm)

    pm.ENABLE_GHOST_PROBE  = enable_pa_loi
    pm.ENABLE_AEB          = enable_aeb
    pm.ENABLE_DATA_LOGGING = True

    from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator

    print(f"\n{'='*65}")
    print(f"  Trial: [{label}]  PA-LOI={enable_pa_loi}  AEB={enable_aeb}")
    print(f"  Frames: {N_FRAMES} | Scene: ghost_experiment (intersection)")
    print(f"{'='*65}")

    sim = GhostProbeSimulator(CONFIG, enable_ghost_probe_defense=enable_pa_loi)
    sim.init_sim()

    # ★ KEY: suppress ghost spawn → no pedestrian will ever appear
    sim.ghost_config  = None
    sim.ghost_spawned = True

    sim.sim_horizon = N_FRAMES
    sim.render      = False
    sim.output_dir  = os.path.join(OUTPUT_DIR, label)
    os.makedirs(sim.output_dir, exist_ok=True)

    # ── Per-frame recorder ──────────────────────────────────────────────────
    frame_log = []

    def patched_run_sim(self):
        from tqdm import tqdm
        from agent import CustomizedAgent, NonReactiveAgent

        print("[NO_GHOST] Running no-ghost efficiency trial...", flush=True)
        self.frames   = []
        self.sim_time = 0.0
        terminated    = False

        for step_idx in tqdm(range(self.sim_horizon)):
            # --- Build observation list ---
            agent_obs = []
            for agent in self.agents:
                if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) \
                        or isinstance(agent, CustomizedAgent):
                    agent_obs.append(agent.observe())

            agent_gt = [a.observe_no_noise() for a in self.agents
                        if (isinstance(a, NonReactiveAgent) and a.is_valid())
                        or isinstance(a, CustomizedAgent)]
            self.frames.append({'agents': agent_gt})

            # --- Record ego speed ---
            ego_agent = next((a for a in self.agents if a.id == 'AV'), None)
            if ego_agent:
                ego_vel = float(ego_agent.state[2]) if len(ego_agent.state) > 2 else 0.0
                frame_log.append({
                    'step':     step_idx,
                    'sim_time': round(self.sim_time, 3),
                    'ego_vel':  round(ego_vel, 4),
                })

            # --- Step agents (mirrors original run_sim logic) ---
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
                                print(f"[NO_GHOST] Planner failure at step {step_idx}", flush=True)
                                terminated = True
                                break
                elif isinstance(agent, NonReactiveAgent):
                    agent.step()
                agent.update_state(self.sim_step)

            self.sim_time += self.sim_step

            if terminated:
                break

        print(f"[NO_GHOST] Done. Steps={step_idx+1}, "
              f"Terminated={'Yes' if terminated else 'No (timeout)'}")

    import types
    sim.run_sim = types.MethodType(patched_run_sim, sim)

    t0 = time.perf_counter()
    sim.run()
    elapsed = time.perf_counter() - t0

    # ── Statistics ──────────────────────────────────────────────────────────
    if frame_log:
        vels        = [r['ego_vel'] for r in frame_log]
        v_avg       = float(np.mean(vels))
        v_min       = float(np.min(vels))
        v_max       = float(np.max(vels))
        slow_frames = sum(1 for v in vels if v < 6.0)   # < 75% of target 8 m/s
        slow_pct    = 100.0 * slow_frames / len(vels)
    else:
        v_avg = v_min = v_max = slow_pct = slow_frames = float('nan')

    print(f"\n  ── [{label}] Statistics ─────────────────────")
    print(f"     Frames    : {len(frame_log)}")
    print(f"     Avg speed : {v_avg:.3f} m/s")
    print(f"     Min speed : {v_min:.3f} m/s")
    print(f"     Max speed : {v_max:.3f} m/s")
    print(f"     Slow (<6m/s): {slow_frames} frames ({slow_pct:.1f}%)")
    print(f"     Wall time : {elapsed:.1f} s")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, f'{label}_frame_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'sim_time', 'ego_vel'])
        writer.writeheader()
        writer.writerows(frame_log)
    print(f"     CSV saved : {csv_path}")

    return {
        'label':    label,
        'n_frames': len(frame_log),
        'v_avg':    v_avg,
        'v_min':    v_min,
        'v_max':    v_max,
        'slow_pct': slow_pct,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = []

    # Trial A: Vanilla MIND — no protection (baseline)
    results.append(run_one_trial('vanilla_mind', enable_pa_loi=False, enable_aeb=False))

    # Trial B: PA-LOI + AEB — no pedestrian appears
    results.append(run_one_trial('paloi_aeb_no_ghost', enable_pa_loi=True, enable_aeb=True))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  EFFICIENCY COMPARISON (No pedestrian / threat-free)")
    print(f"{'='*65}")
    print(f"  {'Config':<26} {'Avg m/s':>8} {'Min m/s':>8} {'Slow%':>6}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['label']:<26} {r['v_avg']:>8.3f} {r['v_min']:>8.3f} {r['slow_pct']:>5.1f}%")
    print(f"{'='*65}")

    if len(results) == 2:
        delta = results[1]['v_avg'] - results[0]['v_avg']
        pct   = 100.0 * delta / (results[0]['v_avg'] + 1e-9)
        print(f"\n  PA-LOI speed impact vs Vanilla: {delta:+.3f} m/s ({pct:+.1f}%)")
        tag = "✅ < 5% — negligible" if abs(pct) < 5.0 else "⚠️  >= 5% — worth noting"
        print(f"  {tag}")

    print(f"\n  Data saved to: {OUTPUT_DIR}/")
