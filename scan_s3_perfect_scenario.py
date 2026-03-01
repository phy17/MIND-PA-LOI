import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tempfile
import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization as ss
from av2.datasets.motion_forecasting.data_schema import ObjectType
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = 'argoverse'
PREFIX = 'datasets/av2/motion-forecasting/val/'
MAX_WORKERS = 32
TARGET_HITS = 2

# Anonymous S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def process_file(key):
    scene_id = key.split('/')[-1].replace('scenario_', '').replace('.parquet', '')
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=True) as tmp:
        try:
            s3.download_file(BUCKET, key, tmp.name)
            scenario = ss.load_argoverse_scenario_parquet(tmp.name)
        except Exception as e:
            return None
            
        # Get AV trajectory
        av_track = next((t for t in scenario.tracks if t.track_id == 'AV'), None)
        if av_track is None: return None
        
        ego_states = av_track.object_states
        if len(ego_states) < 110: return None
        
        ego_traj = np.array([[st.position[0], st.position[1]] for st in ego_states])
        ego_vels = np.array([[st.velocity[0], st.velocity[1]] for st in ego_states])
        ego_speeds = np.linalg.norm(ego_vels, axis=1)
        
        candidates = []
        for track in scenario.tracks:
            if track.track_id == 'AV': continue
            if track.object_type != ObjectType.VEHICLE: continue
            
            agent_states = track.object_states
            if not agent_states: continue
            
            init_v = np.linalg.norm([agent_states[0].velocity[0], agent_states[0].velocity[1]])
            if init_v > 0.5: continue
            
            agent_pos = np.array([agent_states[0].position[0], agent_states[0].position[1]])
            
            distances = np.linalg.norm(ego_traj - agent_pos, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # STRICT REQUIREMENTS
            if 1.5 <= min_dist <= 2.2:
                if min_idx >= 30:
                    av_speed = ego_speeds[min_idx]
                    if av_speed >= 5.0:
                        dist_to_start = sum(np.linalg.norm(ego_traj[i+1] - ego_traj[i]) for i in range(min_idx))
                        candidates.append({
                            'scene_id': scene_id,
                            'track_id': track.track_id,
                            'dist_lat': min_dist,
                            'dist_lon': dist_to_start,
                            'av_speed': av_speed,
                            'idx': min_idx
                        })
                        
        if candidates:
            return candidates
    return None

def main():
    print(f"Scanning S3 incrementally for {TARGET_HITS} perfect scenarios...")
    print("Criteria: VEHICLE occluder, lat=1.5-2.2m, lon_idx>=30, speed>=5.0m/s")
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)
    
    hits = []
    scanned = 0
    
    for page in pages:
        if 'Contents' not in page: continue
        keys = [obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.parquet')]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_file, key): key for key in keys}
            
            for future in as_completed(futures):
                scanned += 1
                if scanned % 100 == 0:
                    print(f"  Scanned {scanned} files so far...")
                    
                res = future.result()
                if res is not None:
                    for c in res:
                        print(f"\n✅ FOUND PERFECT SCENARIO! ({len(hits)+1}/{TARGET_HITS})")
                        print(f"  Scene: {c['scene_id']}")
                        print(f"  Occluder ID: {c['track_id']}")
                        print(f"  Lat Dist: {c['dist_lat']:.2f}m")
                        print(f"  Lon Dist: {c['dist_lon']:.1f}m")
                        print(f"  AV Speed: {c['av_speed']:.1f} m/s")
                        print("-" * 40)
                        hits.append(c)
                
                if len(hits) >= TARGET_HITS:
                    print("\n🎉 Target reached! Found enough scenarios. Exiting.")
                    # Download the dataset files for the best one to our local directory
                    best = hits[0]
                    scene_id = best['scene_id']
                    
                    local_dir = f"/Users/phy/Desktop/MIND/data/{scene_id}"
                    os.makedirs(local_dir, exist_ok=True)
                    print(f"Downloading required files for scene {scene_id} to local data folder...")
                    
                    # Also need to download the map JSON
                    map_key = f"datasets/av2/motion-forecasting/val/{scene_id}/log_map_archive_{scene_id}.json"
                    parquet_key = f"datasets/av2/motion-forecasting/val/{scene_id}/scenario_{scene_id}.parquet"
                    
                    s3.download_file(BUCKET, parquet_key, os.path.join(local_dir, f"scenario_{scene_id}.parquet"))
                    s3.download_file(BUCKET, map_key, os.path.join(local_dir, f"log_map_archive_{scene_id}.json"))
                    print(f"Successfully downloaded to {local_dir}")
                    
                    # Update ghost_experiment_vehicle.json
                    import json
                    config_path = "/Users/phy/Desktop/MIND/configs/ghost_experiment_vehicle.json"
                    with open(config_path, 'r') as f: config = json.load(f)
                    config['seq_id'] = scene_id
                    with open(config_path, 'w') as f: json.dump(config, f, indent=4)
                    print(f"Updated {config_path} with new scene ID.")
                    
                    # Clean shutdown of executor
                    executor.shutdown(wait=False, cancel_futures=True)
                    return

if __name__ == "__main__":
    main()
