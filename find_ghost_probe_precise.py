"""
Augmented Argoverse 2 Ghost Probe Finder (Online)
高精度鬼探头场景筛选器 - 基于几何遮挡计算
"""

import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import os
import math

# 配置 (全量扫描版)
MAX_SCENARIOS = 250000  # 扫描全部验证集
AV_MIN_SPEED = 1.0
PED_MIN_SPEED = 0.5
BLOCKER_WIDTH = 2.0

# ... (get_scenario_list 保持不变)

def get_scenario_list(limit=500):
    """获取场景列表"""
    print(f"获取场景列表（前 {limit} 个）...")
    result = subprocess.run(
        ["s5cmd", "--no-sign-request", "ls", "s3://argoverse/datasets/av2/motion-forecasting/train/"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split('\n')
    scenarios = []
    for line in lines:
        if 'DIR' in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                scenario_id = parts[-1].rstrip('/')
                if scenario_id and len(scenario_id) == 36:
                    scenarios.append(scenario_id)
    return scenarios[:limit]

def geometry_check(av, blocker, ped):
    """
    几何判定 (宽容版)
    """
    p_av = np.array([av['x'], av['y']])
    p_block = np.array([blocker['x'], blocker['y']])
    p_ped = np.array([ped['x'], ped['y']])
    
    dist_av_block = np.linalg.norm(p_av - p_block)
    dist_ped_block = np.linalg.norm(p_ped - p_block)
    
    # 距离放宽
    if dist_av_block < 2.0: return False # 太近了
    if dist_ped_block > 10.0: return False # 行人离车太远

    # 简单遮挡判断：行人和遮挡物相对于 AV 的夹角很小
    vec_block = p_block - p_av
    angle_block = np.arctan2(vec_block[1], vec_block[0])
    
    vec_ped = p_ped - p_av
    angle_ped = np.arctan2(vec_ped[1], vec_ped[0])
    
    # 计算遮挡角 (假设遮挡物宽 2m, 给一点余量 *1.5 -> *2.5 放宽判定)
    angular_half_width = np.arctan((BLOCKER_WIDTH / 2.0) / dist_av_block) * 2.5
    
    angle_diff = abs(angle_ped - angle_block)
    if angle_diff > np.pi: angle_diff = 2*np.pi - angle_diff
    
    # 只要在遮挡角附近就算 (宽容判定)
    if angle_diff < angular_half_width:
        return True
        
    return False

def download_and_analyze(scenario_id):
    s3_path = f"s3://argoverse/datasets/av2/motion-forecasting/train/{scenario_id}/scenario_{scenario_id}.parquet"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, f"scenario_{scenario_id}.parquet")
        result = subprocess.run(
            ["s5cmd", "--no-sign-request", "cp", s3_path, local_path],
            capture_output=True, text=True
        )
        if result.returncode != 0: return None
        try:
            df = pd.read_parquet(local_path)
        except: return None
        
        timesteps = df['timestep'].unique()
        check_steps = timesteps[::10] # 提高采样率：每1秒检测一次
        
        if 'focal_track_id' in df.columns:
            focal_id = df['focal_track_id'].iloc[0]
        else:
            focal_id = 'AV'

        for ts in check_steps:
            frame = df[df['timestep'] == ts]
            
            try:
                av_data = frame[frame['track_id'] == focal_id]
                if av_data.empty: av_data = frame[frame['track_id'] == 'AV']
                if av_data.empty: continue
            except: continue
            
            av_info = {
                'x': av_data['position_x'].values[0],
                'y': av_data['position_y'].values[0],
                'vx': av_data['velocity_x'].values[0],
                'vy': av_data['velocity_y'].values[0]
            }
            if np.hypot(av_info['vx'], av_info['vy']) < AV_MIN_SPEED: continue

            # 2. 找静态遮挡物 (放宽：包含所有 VEHICLE)
            blockers = frame[
                (frame['object_type'].isin(['BUS', 'TRUCK', 'VEHICLE', 'vehicle']))
            ]
            
            valid_blockers = []
            for _, b in blockers.iterrows():
                if np.hypot(b['velocity_x'], b['velocity_y']) < 0.2: # 稍微放宽静止判定
                    valid_blockers.append({'x': b['position_x'], 'y': b['position_y'], 'id': b['track_id'], 'type': b['object_type']})
            
            if not valid_blockers: continue

            # 3. 找动态行人
            peds = frame[frame['object_type'] == 'PEDESTRIAN']
            valid_peds = []
            for _, p in peds.iterrows():
                if np.hypot(p['velocity_x'], p['velocity_y']) > PED_MIN_SPEED:
                    valid_peds.append({'x': p['position_x'], 'y': p['position_y'], 'id': p['track_id']})
            
            if not valid_peds: continue

            # 4. 几何匹配
            for b in valid_blockers:
                dist_av_b = np.hypot(b['x']-av_info['x'], b['y']-av_info['y'])
                if dist_av_b > 40 or dist_av_b < 2: continue
                
                for p in valid_peds:
                    if geometry_check(av_info, b, p):
                        return {
                            'scenario_id': scenario_id,
                            'timestamp': ts,
                            'details': f"遮挡物:{b['type']}, AV速:{np.hypot(av_info['vx'], av_info['vy']):.1f}"
                        }
        return None

def main():
    # ...
    
    scenarios = get_scenario_list(MAX_SCENARIOS)
    print(f"扫描场景池大小: {len(scenarios)}")
    print(f"启动超线程扫描 (线程数: 64)...")
    
    found_count = 0
    import concurrent.futures
    import threading
    lock = threading.Lock()
    
    # 进度计数
    processed_count = 0
    
    with open("ghost_probe_candidates_full.txt", "w") as f, \
         concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        
        # 提交所有任务
        future_to_sid = {executor.submit(download_and_analyze, sid): sid for sid in scenarios}
        
        for future in concurrent.futures.as_completed(future_to_sid):
            sid = future_to_sid[future]
            with lock:
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"\rProgress: {processed_count}/{len(scenarios)} | Found: {found_count}", end="", flush=True)
            
            try:
                res = future.result()
                if res:
                    with lock:
                        found_count += 1
                        print(f"\n✅ FOUND! {sid}")
                        print(f"   Details: {res['details']}")
                        f.write(f"{sid}\n")
                        f.flush()
            except Exception as e:
                pass # 忽略单个任务的失败
    
    print(f"\n\n扫描结束! 共找到 {found_count} 个潜在场景.")
    print("列表已保存至 ghost_probe_candidates.txt")

if __name__ == "__main__":
    main()
