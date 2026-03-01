import sys
import os
import argparse
import json
import subprocess

# Add project root to path
sys.path.insert(0, os.getcwd())

from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator

def run_baseline_only(config_path, output_dir):
    print("="*60)
    print("  RUN BASELINE ONLY (No Defense, Expect Collision)")
    print("="*60 + "\n")
    
    # Initialize Simulator without Defense
    # Enable rendering!
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        cfg['render'] = True
    
    # Override config on the fly if needed, or rely on json
    # (Here we assume config.json has render: true)
    
    sim = GhostProbeSimulator(config_path, enable_ghost_probe_defense=False)
    sim.sim_horizon = 600 # 600 frames as requested
    sim.init_sim()
    sim.output_dir = output_dir
    sim.run()
    
    print("\n[Baseline Log] Collision Count:", len(sim.collision_log))
    if len(sim.collision_log) > 0:
        print("✅ Correct! Collision happened as expected for Baseline.")
    else:
        print("⚠️ Warning: No collision? Check trigger distance.")
        
    print(f"\nResults saved to: {output_dir}")

    # Explicitly call video generation
    try:
        print("Rendering video...")
        cmd = ["python", "scripts/image_to_video.py", "--image_folder", f"{output_dir}/imgs", "--output_file", f"{output_dir}/baseline_crash.mp4"]
        subprocess.run(cmd, check=True)
        print(f"Video saved to: {output_dir}/baseline_crash.mp4")
    except Exception as e:
        print(f"Video rendering failed: {e}")

if __name__ == "__main__":
    run_baseline_only("configs/ghost_experiment.json", "output/ghost_experiment/baseline/")
