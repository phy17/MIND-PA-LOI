"""
PA-LOI 计算效率基准测试
用于论文中的计算效率量化指标
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np
import torch

from planners.mind.utils import get_semantic_risk_sources, calculate_adaptive_corridor

def benchmark_paloi_overhead():
    """测量 PA-LOI 模块的单帧平均计算耗时"""
    
    device = 'cpu'
    N_AGENTS = 20  # 典型场景中的智能体数量
    T_STEPS = 50   # 历史帧数
    
    # 模拟典型输入数据
    trajs_pos = torch.randn(N_AGENTS, T_STEPS, 2, device=device) * 20.0
    trajs_vel = torch.randn(N_AGENTS, T_STEPS, 2, device=device) * 0.1  # 大部分静止
    trajs_ang = torch.randn(N_AGENTS, T_STEPS, device=device)
    
    # 类型 one-hot: [VEHICLE, PED, CYCLIST, MOTORCYCLE, BUS, ...]
    trajs_type = torch.zeros(N_AGENTS, T_STEPS, 10, device=device)
    trajs_type[1:10, :, 0] = 1   # 前9个设为 VEHICLE
    trajs_type[10:15, :, 4] = 1  # 5个设为 BUS
    # 其余为 unknown（不会被选为 occluder）
    
    ego_pos = torch.tensor([0.0, 0.0], device=device)
    ego_heading = torch.tensor(0.0, device=device)
    ego_vel = 5.0
    
    target_lane = np.array([[i*2.0, 0.0] for i in range(20)])  # 简单直线车道
    
    # ===== Warm-up =====
    print("预热中...")
    for _ in range(5):
        get_semantic_risk_sources(
            trajs_pos, trajs_vel, trajs_type, trajs_ang,
            ego_pos=ego_pos, ego_heading=ego_heading, device=device,
            ego_vel=ego_vel, lane_width=3.5, road_width=14.0,
            target_lane=target_lane
        )
    
    # ===== Benchmark: get_semantic_risk_sources =====
    N_RUNS = 200
    print(f"\n=== PA-LOI Risk Source Identification Benchmark ({N_RUNS} runs) ===")
    
    times_risk = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        risk_sources = get_semantic_risk_sources(
            trajs_pos, trajs_vel, trajs_type, trajs_ang,
            ego_pos=ego_pos, ego_heading=ego_heading, device=device,
            ego_vel=ego_vel, lane_width=3.5, road_width=14.0,
            target_lane=target_lane
        )
        t1 = time.perf_counter()
        times_risk.append((t1 - t0) * 1000)  # ms
    
    # ===== Benchmark: calculate_adaptive_corridor =====
    times_corridor = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        d_crit, d_outer = calculate_adaptive_corridor(3.5, 14.0, ego_vel)
        t1 = time.perf_counter()
        times_corridor.append((t1 - t0) * 1000)  # ms
    
    # ===== Benchmark: risk cost evaluation (simulate weight computation) =====
    times_eval = []
    # 模拟 evaluate_traj_tree 中的 risk_cost 计算
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        # 模拟遍历 trajectory nodes 并计算 risk cost
        n_nodes = 8  # 典型的 trajectory tree 节点数
        for _ in range(n_nodes):
            dummy_pos = torch.randn(30, 2, device=device)
            for rs in risk_sources:
                pos = rs['pos']
                cov_inv = torch.linalg.inv(rs['cov'])
                diff = dummy_pos - pos
                mahal = torch.sum(diff @ cov_inv * diff, dim=-1)
                cost = rs['weight'] * torch.exp(-0.5 * mahal)
        t1 = time.perf_counter()
        times_eval.append((t1 - t0) * 1000)
    
    # ===== Results =====
    print("\n" + "="*60)
    print("PA-LOI 计算效率基准测试结果")
    print("="*60)
    
    arr_risk = np.array(times_risk)
    arr_corridor = np.array(times_corridor)
    arr_eval = np.array(times_eval)
    
    print(f"\n[1] Risk Source Identification (get_semantic_risk_sources):")
    print(f"    Mean:   {arr_risk.mean():.3f} ms")
    print(f"    Median: {np.median(arr_risk):.3f} ms")
    print(f"    Std:    {arr_risk.std():.3f} ms")
    print(f"    Max:    {arr_risk.max():.3f} ms")
    print(f"    Agents: {N_AGENTS}")
    
    print(f"\n[2] Adaptive Corridor Calculation:")
    print(f"    Mean:   {arr_corridor.mean():.4f} ms")
    print(f"    Median: {np.median(arr_corridor):.4f} ms")
    
    print(f"\n[3] Risk Cost Evaluation (per trajectory tree):")
    print(f"    Mean:   {arr_eval.mean():.3f} ms")
    print(f"    Median: {np.median(arr_eval):.3f} ms")
    
    total_mean = arr_risk.mean() + arr_corridor.mean() + arr_eval.mean()
    total_median = np.median(arr_risk) + np.median(arr_corridor) + np.median(arr_eval)
    
    print(f"\n[TOTAL] PA-LOI Per-Frame Overhead:")
    print(f"    Mean:   {total_mean:.3f} ms")
    print(f"    Median: {total_median:.3f} ms")
    print(f"    Budget: {100.0} ms (10 Hz planning)")
    print(f"    Overhead: {total_mean / 100.0 * 100:.1f}% of planning budget")
    
    print(f"\n[INFO] Risk sources found: {len(risk_sources)}")
    
    return {
        'risk_identification_ms': float(arr_risk.mean()),
        'corridor_calc_ms': float(arr_corridor.mean()),
        'risk_eval_ms': float(arr_eval.mean()),
        'total_overhead_ms': float(total_mean),
        'n_agents': N_AGENTS,
        'n_runs': N_RUNS,
    }

if __name__ == '__main__':
    results = benchmark_paloi_overhead()
    print(f"\n{'='*60}")
    print(f"Summary JSON: {results}")
