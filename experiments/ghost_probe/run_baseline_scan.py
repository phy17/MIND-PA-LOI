#!/usr/bin/env python3
"""
Baseline Scan: Vanilla MIND + AEB Only
同样的 6 个触发距离，跑两组基线对比实验。
"""

import sys, os, json, csv, time
import importlib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TRIGGER_DISTANCES = [3.0, 3.5, 4.0, 4.3, 4.5, 5.0]
VEHICLE_HALF_LENGTH = 2.25
SIM_HORIZON = 550

OUTPUT_BASE = "output/baseline_scan"

def run_single(trigger_dist, exp_name, enable_defense, enable_aeb):
    import planners.mind.planner as pm
    importlib.reload(pm)
    pm.ENABLE_GHOST_PROBE = enable_defense
    pm.ENABLE_AEB = enable_aeb
    pm.ENABLE_DATA_LOGGING = False

    from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator
    sim = GhostProbeSimulator('configs/ghost_experiment.json',
                               enable_ghost_probe_defense=enable_defense)
    sim.init_sim()
    sim.strict_trigger_dist = trigger_dist
    sim.output_dir = os.path.join(OUTPUT_BASE, exp_name)
    sim.sim_horizon = SIM_HORIZON
    sim.render = False
    os.makedirs(sim.output_dir, exist_ok=True)

    start = time.time()
    sim.run()
    elapsed = time.time() - start

    collisions = len(sim.collision_log)
    col_vel = sim.collision_log[0]['ego_vel'] if sim.collision_log else 0.0

    return {
        'trigger_dist_center': trigger_dist,
        'trigger_dist_bumper': round(trigger_dist - VEHICLE_HALF_LENGTH, 2),
        'system': exp_name.split('_d')[0],
        'collisions': collisions,
        'collision_vel_ms': round(col_vel, 3),
        'elapsed_s': round(elapsed, 1),
    }

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_results = []

    configs = [
        ("vanilla",  False, False),  # No PA-LOI, No AEB
        ("aeb_only", False, True),   # No PA-LOI, AEB enabled
    ]

    for sys_name, defense, aeb in configs:
        print(f"\n{'='*60}")
        print(f"  SYSTEM: {sys_name.upper()}")
        print(f"  PA-LOI={defense}, AEB={aeb}")
        print(f"{'='*60}")

        for dist in TRIGGER_DISTANCES:
            bumper = round(dist - VEHICLE_HALF_LENGTH, 2)
            exp_name = f"{sys_name}_d{dist:.1f}"
            print(f"\n--- {exp_name} (bumper={bumper}m) ---")
            result = run_single(dist, exp_name, defense, aeb)
            all_results.append(result)
            status = "✅ SAFE" if result['collisions'] == 0 else f"❌ {result['collision_vel_ms']:.2f} m/s"
            print(f"  Result: {status} ({result['elapsed_s']}s)")

    # 汇总
    print("\n\n" + "="*80)
    print("  BASELINE RESULTS SUMMARY")
    print("="*80)
    print(f"{'System':<12}{'D_center':<10}{'D_bumper':<10}{'Collision':<10}{'Vel(m/s)':<12}")
    print("-"*54)
    for r in all_results:
        col = "YES" if r['collisions'] > 0 else "NO"
        vel = f"{r['collision_vel_ms']:.2f}" if r['collisions'] > 0 else "N/A"
        print(f"{r['system']:<12}{r['trigger_dist_center']:<10}{r['trigger_dist_bumper']:<10}{col:<10}{vel:<12}")

    # 保存
    csv_path = os.path.join(OUTPUT_BASE, "baseline_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV: {csv_path}")

    json_path = os.path.join(OUTPUT_BASE, "baseline_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON: {json_path}")

if __name__ == "__main__":
    main()
