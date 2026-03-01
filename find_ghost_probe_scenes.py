"""
Ghost Probe Scene Finder
筛选 Argoverse 2 中可能存在"鬼探头"场景的脚本

条件：
1. 有静止的大型车辆（Bus/Vehicle，速度 < 0.5 m/s）
2. 该车辆在 AV 的前方
3. 该车辆附近有行人
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/phy/Desktop/MIND/data")

def analyze_scenario(scenario_id):
    """分析单个场景，判断是否可能存在鬼探头"""
    parquet_path = DATA_DIR / scenario_id / f"scenario_{scenario_id}.parquet"
    
    if not parquet_path.exists():
        return None
    
    df = pd.read_parquet(parquet_path)
    
    # 获取最后一帧的数据
    last_timestep = df['timestep'].max()
    last_frame = df[df['timestep'] == last_timestep]
    
    # 找到 AV
    av_data = last_frame[last_frame['track_id'] == 'AV']
    if av_data.empty:
        # 尝试 focal_track_id
        focal_id = df[df['focal_track_id'].notna()]['focal_track_id'].iloc[0] if 'focal_track_id' in df.columns else None
        if focal_id:
            av_data = last_frame[last_frame['track_id'] == focal_id]
    
    if av_data.empty:
        return None
    
    av_x = av_data['position_x'].values[0]
    av_y = av_data['position_y'].values[0]
    
    # 找静止的大型车辆
    vehicles = last_frame[last_frame['object_type'].isin(['vehicle', 'bus', 'VEHICLE', 'BUS'])]
    
    # 计算速度
    stationary_vehicles = []
    for _, veh in vehicles.iterrows():
        vx = veh.get('velocity_x', 0) or 0
        vy = veh.get('velocity_y', 0) or 0
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed < 0.5:  # 静止
            # 计算距离 AV 的距离
            dist = np.sqrt((veh['position_x'] - av_x)**2 + (veh['position_y'] - av_y)**2)
            if 5 < dist < 50:  # 在合理范围内
                stationary_vehicles.append({
                    'track_id': veh['track_id'],
                    'type': veh['object_type'],
                    'distance': dist,
                    'x': veh['position_x'],
                    'y': veh['position_y']
                })
    
    # 找行人
    pedestrians = last_frame[last_frame['object_type'].isin(['pedestrian', 'PEDESTRIAN'])]
    
    # 检查是否有行人在静止车辆附近
    ghost_probe_risk = False
    risk_details = []
    
    for veh in stationary_vehicles:
        for _, ped in pedestrians.iterrows():
            ped_dist_to_veh = np.sqrt((ped['position_x'] - veh['x'])**2 + (ped['position_y'] - veh['y'])**2)
            if ped_dist_to_veh < 10:  # 行人在车辆 10 米范围内
                ghost_probe_risk = True
                risk_details.append({
                    'vehicle': veh['track_id'],
                    'pedestrian': ped['track_id'],
                    'ped_dist_to_vehicle': ped_dist_to_veh
                })
    
    return {
        'scenario_id': scenario_id,
        'has_stationary_vehicles': len(stationary_vehicles) > 0,
        'stationary_vehicle_count': len(stationary_vehicles),
        'pedestrian_count': len(pedestrians),
        'ghost_probe_risk': ghost_probe_risk,
        'risk_details': risk_details
    }


def main():
    print("=" * 60)
    print("Ghost Probe Scene Finder")
    print("=" * 60)
    
    scenarios = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n找到 {len(scenarios)} 个场景")
    print("-" * 60)
    
    results = []
    for scenario_id in scenarios:
        result = analyze_scenario(scenario_id)
        if result:
            results.append(result)
            
            status = ""
            if result['ghost_probe_risk']:
                status = "🔴 高风险（有行人在静止车辆附近）"
            elif result['has_stationary_vehicles']:
                status = "🟡 存在静止车辆"
            else:
                status = "🟢 低风险"
            
            print(f"\n{scenario_id[:8]}...:")
            print(f"  静止车辆: {result['stationary_vehicle_count']}")
            print(f"  行人数量: {result['pedestrian_count']}")
            print(f"  风险评估: {status}")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    high_risk = [r for r in results if r['ghost_probe_risk']]
    medium_risk = [r for r in results if r['has_stationary_vehicles'] and not r['ghost_probe_risk']]
    
    print(f"🔴 高风险场景: {len(high_risk)} 个")
    for r in high_risk:
        print(f"   - {r['scenario_id']}")
    
    print(f"🟡 中风险场景: {len(medium_risk)} 个")
    for r in medium_risk:
        print(f"   - {r['scenario_id']}")


if __name__ == "__main__":
    main()
