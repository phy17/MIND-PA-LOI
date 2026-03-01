import os
import glob
import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization as ss
from av2.datasets.motion_forecasting.data_schema import ObjectType

DATA_DIR = "/Users/phy/Desktop/MIND/data/"

def find_perfect_scenarios():
    print("Searching for perfect generalization scenarios...")
    print("Requirements:")
    print("1. Occluder type: VEHICLE")
    print("2. Lateral distance: 1.5m to 2.2m")
    print("3. Longitudinal distance: >30m from AV start")
    print("4. AV speed at passing: >5.0 m/s")
    print("-" * 50)
    
    scenario_dirs = glob.glob(os.path.join(DATA_DIR, "*"))
    best_candidates = []
    
    for s_dir in scenario_dirs:
        if not os.path.isdir(s_dir): continue
        scene_id = os.path.basename(s_dir)
        parquet_path = os.path.join(s_dir, f"scenario_{scene_id}.parquet")
        if not os.path.exists(parquet_path): continue
        
        try:
            scenario = ss.load_argoverse_scenario_parquet(parquet_path)
        except:
            continue
            
        # Get AV trajectory
        av_track = next((t for t in scenario.tracks if t.track_id == 'AV'), None)
        if av_track is None: continue
        
        ego_states = av_track.object_states
        if len(ego_states) < 110: continue
        
        ego_traj = np.array([[st.position[0], st.position[1]] for st in ego_states])
        ego_vels = np.array([[st.velocity[0], st.velocity[1]] for st in ego_states])
        ego_speeds = np.linalg.norm(ego_vels, axis=1)
        
        for track in scenario.tracks:
            if track.track_id == 'AV': continue
            if track.object_type != ObjectType.VEHICLE: continue
            
            # Check if static
            agent_states = track.object_states
            if not agent_states: continue
            init_v = np.linalg.norm([agent_states[0].velocity[0], agent_states[0].velocity[1]])
            if init_v > 0.5: continue
            
            agent_pos = np.array([agent_states[0].position[0], agent_states[0].position[1]])
            
            # Find closest point on ego trajectory
            distances = np.linalg.norm(ego_traj - agent_pos, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # Lateral distance check (1.8m to 3.5m)
            if 1.8 <= min_dist <= 3.5:
                # Longitudinal check: must be far enough from AV start (e.g. at least index 20 @ 10Hz)
                if min_idx < 20: continue
                
                # Check AV speed
                av_speed = ego_speeds[min_idx]
                if av_speed > 3.5:
                    dist_to_start = sum(np.linalg.norm(ego_traj[i+1] - ego_traj[i]) for i in range(min_idx))
                    best_candidates.append({
                        'scene_id': scene_id,
                        'track_id': track.track_id,
                        'dist_lat': min_dist,
                        'dist_lon': dist_to_start,
                        'av_speed': av_speed,
                        'idx': min_idx
                    })
                    print(f"Candidate found: Scene {scene_id}")
                    print(f"  Track ID: {track.track_id}")
                    print(f"  Lat Dist: {min_dist:.2f}m")
                    print(f"  Lon Dist: {dist_to_start:.1f}m (traj idx {min_idx})")
                    print(f"  AV Speed: {av_speed:.1f} m/s")
                    print("-" * 30)

    print(f"Found {len(best_candidates)} perfect candidates.")
    
    # Sort by how close lateral distance is to 1.8m (sweet spot)
    best_candidates.sort(key=lambda x: abs(x['dist_lat'] - 1.8))
    
    if best_candidates:
        print("\nTOP RECOMMENDATION:")
        print(f"Scene: {best_candidates[0]['scene_id']}")
        print(f"Lat Dist: {best_candidates[0]['dist_lat']:.2f}m")
        print(f"Lon Dist: {best_candidates[0]['dist_lon']:.1f}m")
        print(f"AV Speed: {best_candidates[0]['av_speed']:.1f} m/s")

if __name__ == "__main__":
    find_perfect_scenarios()
