#!/usr/bin/env python3
"""
Quick single test: PA-LOI + AEB on new Vehicle occlusion scenario
Scene: 00010486-9a07-48ae-b493-cf4545855937
Trigger distance: 5.0m (center), render enabled
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import planners.mind.planner as pm
importlib.reload(pm)

# Enable PA-LOI defense + AEB
pm.ENABLE_GHOST_PROBE = True
pm.ENABLE_AEB = True
pm.ENABLE_DATA_LOGGING = False

from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator

CONFIG = 'configs/ghost_experiment_vehicle.json'
TRIGGER_DIST = 5.0
OUTPUT_DIR = 'output/ghost_vehicle_test'

print("=" * 70)
print(f"  Single Test: PA-LOI + AEB on VEHICLE occlusion scenario")
print(f"  Scene: 0168106d-6aac-4b54-adec-fb9996040418")
print(f"  Trigger distance: {TRIGGER_DIST}m (center)")
print(f"  Defense: PA-LOI + AEB")
print("=" * 70)

sim = GhostProbeSimulator(CONFIG, enable_ghost_probe_defense=True)
sim.init_sim()
sim.strict_trigger_dist = TRIGGER_DIST
sim.output_dir = OUTPUT_DIR
sim.sim_horizon = 500
sim.render = True  # 渲染图片

os.makedirs(OUTPUT_DIR, exist_ok=True)

sim.run()

# Results
collisions = len(sim.collision_log)
if collisions > 0:
    vel = sim.collision_log[0]['ego_vel']
    print(f"\n❌ COLLISION! Impact velocity: {vel:.2f} m/s")
else:
    print(f"\n✅ SAFE! No collision occurred.")

print(f"\nGhost spawned: {sim.ghost_spawned}")
print(f"Output saved to: {OUTPUT_DIR}")
print("Done!")
