#!/usr/bin/env python3
"""
Ghost Probe Distance Scan - 论文数据生成脚本 (针对新 VEHICLE 泛化场景)

扫描不同触发距离，收集 PA-LOI + AEB 系统的完整实验数据。
输出: CSV 表格 + 终端汇总，可直接用于论文。
"""

import sys, os, json, csv, time
import importlib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ============================================================
# 实验参数
# ============================================================
TRIGGER_DISTANCES = [3.0, 3.5, 4.0, 4.5, 5.0]  # 质心距离 (m), 我们知道5.0是安全的，往下测临界点
VEHICLE_HALF_LENGTH = 2.25  # 车长4.5m的一半，用于算车头距离
SIM_HORIZON = 500
RENDER = False  # 批量跑不渲染，省时间

OUTPUT_BASE = "output/distance_scan_vehicle_mind_only"

def run_single_experiment(trigger_dist, exp_name, enable_defense, enable_aeb):
    """跑单次实验，返回结果字典"""
    import planners.mind.planner as pm
    importlib.reload(pm)
    
    pm.ENABLE_GHOST_PROBE = enable_defense
    pm.ENABLE_AEB = enable_aeb
    pm.ENABLE_DATA_LOGGING = False  # 批量跑不写详细日志
    
    from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator
    
    # 【注意】使用新场景的 config
    sim = GhostProbeSimulator('configs/ghost_experiment_vehicle.json', 
                               enable_ghost_probe_defense=enable_defense)
    sim.init_sim()
    sim.strict_trigger_dist = trigger_dist
    sim.output_dir = os.path.join(OUTPUT_BASE, exp_name)
    sim.sim_horizon = SIM_HORIZON
    sim.render = RENDER
    
    os.makedirs(sim.output_dir, exist_ok=True)
    
    start_time = time.time()
    planner_failed = False
    try:
        sim.run()
    except Exception as e:
        print(f"Error during simulation: {e}")
        planner_failed = True
    elapsed = time.time() - start_time
    
    # 收集数据
    collisions = len(sim.collision_log)
    if planner_failed:
        collisions = max(1, collisions)
    collision_vel = sim.collision_log[0]['ego_vel'] if sim.collision_log else 0.0
    collision_time = sim.collision_log[0]['timestamp'] if sim.collision_log else None
    
    result = {
        'trigger_dist_center': trigger_dist,
        'trigger_dist_bumper': round(trigger_dist - VEHICLE_HALF_LENGTH, 2),
        'system': exp_name.split('_')[0],
        'collisions': collisions,
        'collision_vel_ms': round(collision_vel, 3),
        'collision_time_s': round(collision_time, 3) if collision_time else None,
        'ghost_spawned': sim.ghost_spawned,
        'elapsed_s': round(elapsed, 1),
    }
    
    return result


def main():
    print("=" * 70)
    print("  Ghost Probe Distance Scan - VEHICLE GENERALIZATION SCENARIO")
    print(f"  触发距离: {TRIGGER_DISTANCES}")
    print(f"  车头距离: {[round(d - VEHICLE_HALF_LENGTH, 2) for d in TRIGGER_DISTANCES]}")
    print(f"  系统: 原版 MIND (无 PA-LOI, 无 AEB)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    all_results = []
    
    for dist in TRIGGER_DISTANCES:
        bumper_dist = round(dist - VEHICLE_HALF_LENGTH, 2)
        print(f"\n{'='*60}")
        print(f"  Distance: {dist}m (center) / {bumper_dist}m (bumper)")
        print(f"{'='*60}")
        
        # Original MIND
        exp_name = f"mind_only_d{dist:.1f}"
        print(f"\n--- Running: {exp_name} ---")
        result = run_single_experiment(dist, exp_name, 
                                        enable_defense=False, enable_aeb=False)
        all_results.append(result)
        
        status = "✅ SAFE" if result['collisions'] == 0 else f"❌ COLLISION @ {result['collision_vel_ms']:.2f} m/s"
        print(f"  Result: {status} ({result['elapsed_s']}s)")
    
    # ============================================================
    # 输出汇总表格
    # ============================================================
    print("\n\n")
    print("=" * 80)
    print("  RESULTS SUMMARY - PA-LOI + AEB (VEHICLE GENERALIZATION)")
    print("=" * 80)
    
    header = f"{'D_center(m)':<14}{'D_bumper(m)':<14}{'Collision':<12}{'Impact V(m/s)':<16}{'Status':<10}"
    print(header)
    print("-" * 66)
    
    for r in all_results:
        col_str = "YES" if r['collisions'] > 0 else "NO"
        vel_str = f"{r['collision_vel_ms']:.2f}" if r['collisions'] > 0 else "N/A"
        status = "❌" if r['collisions'] > 0 else "✅"
        print(f"{r['trigger_dist_center']:<14}{r['trigger_dist_bumper']:<14}"
              f"{col_str:<12}{vel_str:<16}{status:<10}")
    
    print("=" * 80)
    
    # 保存 CSV
    csv_path = os.path.join(OUTPUT_BASE, "scan_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved to: {csv_path}")
    
    # 保存 JSON
    json_path = os.path.join(OUTPUT_BASE, "scan_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {json_path}")
    
    # 找临界点
    safe_dists = [r['trigger_dist_bumper'] for r in all_results if r['collisions'] == 0]
    crash_dists = [r['trigger_dist_bumper'] for r in all_results if r['collisions'] > 0]
    
    if safe_dists and crash_dists:
        critical = min(safe_dists)
        print(f"\n🔑 PA-LOI+AEB 安全临界点 (车头距离): {critical}m")
        print(f"   低于此距离: 碰撞 | 高于此距离: 安全避碰")
    elif not crash_dists:
        print(f"\n🎉 PA-LOI+AEB 在所有测试距离下均安全避碰！")
    else:
        print(f"\n⚠️ PA-LOI+AEB 在所有测试距离下均发生碰撞！")


if __name__ == "__main__":
    main()
