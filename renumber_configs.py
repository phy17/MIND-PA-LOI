import os
import json
import glob
import shutil

data_dir = "data"
config_dir = "configs"
backup_dir = "configs/backup"

# Ensure backup dir exists
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# 1. Get all seq_ids from data directory
if not os.path.exists(data_dir):
    print("Data directory not found!")
    exit(1)

seq_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"Found {len(seq_ids)} sequences in data/.")

# 2. Scan existing configs to preserve settings (camera, agents, etc.)
seq_to_config = {}
all_configs = glob.glob(os.path.join(config_dir, "*.json"))

# Special priority for the correctly configured ghost probe scene
# We know 'run_scenario_002_ghost_hard.json' has the correct config for '0007df76...'
priority_keywords = ['ghost', 'new', 'hard', 'demo_4'] 

for cfg_path in all_configs:
    filename = os.path.basename(cfg_path)
    if filename.startswith('backup'): continue
    
    try:
        with open(cfg_path, 'r') as f:
            data = json.load(f)
            sid = data.get('seq_id')
            if sid and sid in seq_ids:
                # If this sequence is not yet mapped, or if the current file has a "priority keyword" 
                # and the existing one didn't, overwrite it.
                if sid not in seq_to_config:
                    seq_to_config[sid] = data
                else:
                    # heuristic to pick the "better" configured one
                    current_score = sum(1 for k in priority_keywords if k in filename)
                    # We don't have the old filename easily available to compare scores directly without storing more info
                    # So let's just store all candidates and pick best later
                    pass 
    except:
        pass

# Re-scan to strictly pick the best config for each seq_id
for sid in seq_ids:
    best_cfg = None
    best_score = -1
    
    for cfg_path in all_configs:
        try:
            with open(cfg_path, 'r') as f:
                data = json.load(f)
                if data.get('seq_id') == sid:
                    score = 0
                    fname = os.path.basename(cfg_path)
                    if 'ghost' in fname: score += 10
                    if 'new' in fname: score += 5
                    if 'demo_4' in fname and sid == '0007df76-c9a2-47aa-83bf-3b2b414109c9': score += 20 # The target one
                    if 'run_scenario' in fname: score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_cfg = data
        except:
            continue
            
    if best_cfg:
        seq_to_config[sid] = best_cfg

# 3. Create 1.json ... 9.json
# Load a fallback template just in case
default_template = {}
if '0007df76-c9a2-47aa-83bf-3b2b414109c9' in seq_to_config:
    default_template = seq_to_config['0007df76-c9a2-47aa-83bf-3b2b414109c9']
else:
    # Use first available
    if seq_to_config:
        default_template = list(seq_to_config.values())[0]

# Move old configs to backup
for cfg in all_configs:
    shutil.move(cfg, os.path.join(backup_dir, os.path.basename(cfg)))
print(f"Moved old configs to {backup_dir}")

mapping_info = []

for idx, sid in enumerate(seq_ids):
    new_name = f"{idx + 1}.json"
    new_path = os.path.join(config_dir, new_name)
    
    if sid in seq_to_config:
        cfg_data = seq_to_config[sid]
    else:
        cfg_data = default_template.copy()
        cfg_data['seq_id'] = sid
    
    # Update common fields for consistency
    cfg_data['sim_name'] = f"scenario_{idx + 1}"
    cfg_data['output_dir'] = f"output/scenario_{idx + 1}/"
    
    with open(new_path, 'w') as f:
        json.dump(cfg_data, f, indent=4)
        
    desc = f"{new_name} -> Seq: {sid}..."
    if sid == "0007df76-c9a2-47aa-83bf-3b2b414109c9":
        desc += " [🔥 GHOST PROBE / 大巴鬼探头]"
    mapping_info.append(desc)

print("Renaming Complete. New Mapping:")
print("\n".join(mapping_info))
