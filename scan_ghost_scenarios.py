#!/usr/bin/env python3
"""
Ghost Probe Scenario Scanner
=============================
扫描本地所有 Argoverse 2 场景，寻找适合做鬼探头泛化测试的场景。

筛选条件：
1. 场景中存在静止（speed < 0.5 m/s）的车辆（VEHICLE 或 BUS）
2. 该静止车辆距离 AV（自车）的 GT 轨迹在 2.0 ~ 6.0 米之间（旁边但不挡路）
3. 优先寻找 VEHICLE 类型（小轿车）的遮挡物（与当前大巴场景形成对比）

输出：每个场景的遮挡物候选列表，按距离排序
"""

import os
import sys
import numpy as np
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory

DATA_DIR = "data"
CURRENT_SCENE = "f4eaa49a-74a1-4829-81b2-052a650878c3"  # 当前大巴场景（用于标记）

def scan_scenario(seq_id):
    """扫描单个场景，返回所有合适的遮挡物候选"""
    seq_path = os.path.join(DATA_DIR, seq_id)
    parquet_path = Path(seq_path) / f"scenario_{seq_id}.parquet"
    
    if not parquet_path.exists():
        return None, "parquet not found"
    
    try:
        scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
    except Exception as e:
        return None, f"load error: {e}"
    
    # 找到 AV（自车）的轨迹
    av_track = None
    all_tracks = list(scenario.tracks)
    for track in all_tracks:
        if track.category == TrackCategory.FOCAL_TRACK:
            av_track = track
            break
    
    if av_track is None:
        return None, "no focal track (AV)"
    
    # 提取 AV 的位置轨迹
    av_positions = []
    for state in av_track.object_states:
        av_positions.append([state.position[0], state.position[1]])
    av_traj = np.array(av_positions)
    
    if len(av_traj) < 20:
        return None, f"AV trajectory too short ({len(av_traj)} points)"
    
    # AV 的运动范围 & 平均速度
    av_total_dist = np.sum(np.linalg.norm(np.diff(av_traj, axis=0), axis=1))
    av_avg_speed = av_total_dist / (len(av_traj) * 0.1)  # 10Hz
    
    candidates = []
    
    for track in all_tracks:
        if track.track_id == av_track.track_id:
            continue
        
        # 类型映射
        obj_type = track.object_type
        type_name = obj_type.name if hasattr(obj_type, 'name') else str(obj_type)
        
        # 只看车辆类型
        if obj_type not in [ObjectType.VEHICLE, ObjectType.BUS]:
            continue
        
        # 提取该 agent 的轨迹
        agent_positions = []
        agent_velocities = []
        for state in track.object_states:
            agent_positions.append([state.position[0], state.position[1]])
            agent_velocities.append(np.sqrt(state.velocity[0]**2 + state.velocity[1]**2))
        
        if len(agent_positions) < 5:
            continue
        
        agent_pos = np.array(agent_positions)
        agent_vel = np.array(agent_velocities)
        
        # 检查是否静止（全程平均速度 < 0.5 m/s）
        avg_speed = np.mean(agent_vel)
        max_speed = np.max(agent_vel)
        if avg_speed > 0.5:
            continue
        
        # 计算该 agent 与 AV 轨迹的最近距离
        # 取 agent 的中间位置（最稳定的位置采样）
        mid_idx = len(agent_pos) // 2
        agent_center = agent_pos[mid_idx]
        
        # 计算与 AV 轨迹上每一个点的距离
        distances = np.linalg.norm(av_traj - agent_center, axis=1)
        min_dist = np.min(distances)
        min_dist_idx = np.argmin(distances)
        
        # 该遮挡物对应的 AV 轨迹位置在 AV 整条路线的哪个百分比处
        progress_pct = min_dist_idx / len(av_traj) * 100
        
        # 筛选：距离 AV 轨迹在 2.0m ~ 8.0m 之间
        if min_dist < 1.5 or min_dist > 8.0:
            continue
        
        # 计算该遮挡物附近 AV 的大致速度（如果 AV 在那个点接近静止就不太好）
        nearby_start = max(0, min_dist_idx - 5)
        nearby_end = min(len(av_traj) - 1, min_dist_idx + 5)
        if nearby_end > nearby_start:
            nearby_dist = np.linalg.norm(av_traj[nearby_end] - av_traj[nearby_start])
            nearby_time = (nearby_end - nearby_start) * 0.1
            av_speed_near = nearby_dist / nearby_time if nearby_time > 0 else 0
        else:
            av_speed_near = 0
        
        # 计算遮挡物的大致朝向（用于判断是纵向还是横向停放）
        if len(agent_pos) > 1:
            heading_vec = agent_pos[-1] - agent_pos[0]
            heading_len = np.linalg.norm(heading_vec)
        else:
            heading_len = 0
        
        candidates.append({
            'track_id': track.track_id,
            'type': type_name,
            'position': agent_center.tolist(),
            'avg_speed': round(avg_speed, 3),
            'max_speed': round(max_speed, 3),
            'dist_to_av_traj': round(min_dist, 2),
            'av_traj_idx': min_dist_idx,
            'av_progress_pct': round(progress_pct, 1),
            'av_speed_nearby': round(av_speed_near, 2),
            'n_frames': len(agent_pos),
        })
    
    # 按距离排序
    candidates.sort(key=lambda x: x['dist_to_av_traj'])
    
    scene_info = {
        'seq_id': seq_id,
        'av_traj_length': len(av_traj),
        'av_total_dist_m': round(av_total_dist, 1),
        'av_avg_speed_ms': round(av_avg_speed, 2),
        'n_total_tracks': len(all_tracks),
        'is_current_scene': seq_id == CURRENT_SCENE,
    }
    
    return candidates, scene_info


def main():
    print("=" * 80)
    print("  Ghost Probe Scenario Scanner - 鬼探头泛化场景扫描器")
    print("=" * 80)
    
    # 列出所有场景
    scene_dirs = sorted([d for d in os.listdir(DATA_DIR) 
                        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')])
    
    print(f"\n找到 {len(scene_dirs)} 个 Argoverse 2 场景\n")
    
    best_candidates = []
    
    for seq_id in scene_dirs:
        candidates, info = scan_scenario(seq_id)
        
        if candidates is None:
            print(f"  ❌ {seq_id[:12]}... : {info}")
            continue
        
        is_current = "⭐ [当前场景]" if info['is_current_scene'] else ""
        print(f"\n{'='*70}")
        print(f"  场景: {seq_id} {is_current}")
        print(f"  AV轨迹: {info['av_traj_length']} 帧, 总行驶 {info['av_total_dist_m']}m, 均速 {info['av_avg_speed_ms']} m/s")
        print(f"  总交通参与者数: {info['n_total_tracks']}")
        print(f"  找到 {len(candidates)} 个潜在遮挡物:")
        
        if len(candidates) == 0:
            print(f"    (无合适的静止车辆)")
            continue
        
        for i, c in enumerate(candidates):
            quality = "🟢 优秀" if 2.5 <= c['dist_to_av_traj'] <= 5.0 and c['av_speed_nearby'] > 2.0 else \
                      "🟡 可用" if c['av_speed_nearby'] > 1.0 else "🔴 偏弱"
            
            # 特别标注非大巴类型（我们最想要的）
            type_emoji = "🚗" if c['type'] in ['VEHICLE'] else "🚌" if c['type'] in ['BUS'] else "🚛"
            
            print(f"    {i+1}. {type_emoji} {c['type']:15s} | 距AV轨迹 {c['dist_to_av_traj']:5.2f}m "
                  f"| AV附近速度 {c['av_speed_nearby']:4.1f}m/s "
                  f"| 轨迹{c['av_progress_pct']:5.1f}% | {quality}")
            
            # 收集适合泛化的候选（非当前场景 + VEHICLE 类型 + 合理距离）
            if not info['is_current_scene'] and c['type'] in ['VEHICLE'] and \
               2.0 <= c['dist_to_av_traj'] <= 6.0 and c['av_speed_nearby'] > 1.5:
                best_candidates.append({
                    **c,
                    'seq_id': seq_id,
                    'av_avg_speed': info['av_avg_speed_ms'],
                })
    
    # 最终推荐
    print("\n\n" + "=" * 80)
    print("  🏆 最佳泛化测试场景推荐（非当前场景 + 小轿车遮挡 + AV有速度）")
    print("=" * 80)
    
    if not best_candidates:
        print("\n  ⚠️ 未找到完美的小轿车泛化场景！")
        print("  建议：扩展筛选条件，或接受大巴/卡车类型作为第二个测试场景")
        
        # 放宽条件：接受所有类型
        print("\n  === 放宽条件后的候选（所有车辆类型，所有场景）=== ")
        relaxed = []
        for seq_id in scene_dirs:
            if seq_id == CURRENT_SCENE:
                continue
            candidates, info = scan_scenario(seq_id)
            if candidates:
                for c in candidates:
                    if c['av_speed_nearby'] > 1.0 and 2.0 <= c['dist_to_av_traj'] <= 7.0:
                        relaxed.append({**c, 'seq_id': seq_id, 'av_avg_speed': info['av_avg_speed_ms']})
        
        relaxed.sort(key=lambda x: (-x['av_speed_nearby'], x['dist_to_av_traj']))
        for i, r in enumerate(relaxed[:5]):
            print(f"    {i+1}. 场景 {r['seq_id'][:12]}... | {r['type']:10s} "
                  f"| 距AV {r['dist_to_av_traj']:.1f}m | AV速度 {r['av_speed_nearby']:.1f}m/s")
    else:
        best_candidates.sort(key=lambda x: (-x['av_speed_nearby'], x['dist_to_av_traj']))
        for i, b in enumerate(best_candidates[:5]):
            print(f"\n  #{i+1} 场景: {b['seq_id']}")
            print(f"      遮挡物: {b['type']} (track_id: {b['track_id']})")
            print(f"      距AV轨迹: {b['dist_to_av_traj']:.2f}m")
            print(f"      AV附近速度: {b['av_speed_nearby']:.1f} m/s")
            print(f"      AV均速: {b['av_avg_speed']:.1f} m/s")
            print(f"      进度: 轨迹 {b['av_progress_pct']:.1f}%")

    print("\n扫描完成！")


if __name__ == "__main__":
    main()
