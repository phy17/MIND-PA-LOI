import json
import os

SCENARIOS = [
    "00010486-9a07-48ae-b493-cf4545855937",
    "00062a32-8d6d-4449-9948-6fedac67bfcd",
    "0006ca28-fcbb-4ae2-9d9e-951fa3b41c1c",
    "0007df76-c9a2-47aa-83bf-3b2b414109c9",
    "0015197d-b916-43b6-bcaa-8a7d90d7b87d"
]

TEMPLATE = {
    "sim_name": "demo_new_X",
    "seq_id": "PLACEHOLDER",
    "output_dir": "output/demo_new_X/",
    "num_threads": 16,
    "render": True,
    "render_config": {
        "mode": "follow",  # 使用跟随模式，避免手动找坐标
        "camera_position": {
            "x": 0,
            "y": 0,
            "yaw": 0,
            "elev": 90
        }
    },
    "cl_agents": [
        {
            "id": "AV",
            "enable_timestep": 4.0,
            "semantic_lane": -1,
            "target_velocity": 8,
            "agent": "agent:MINDAgent",
            "planner_config": "planners/mind/configs/demo_1.json"
        }
    ],
    "use_cuda": False
}

os.makedirs("configs", exist_ok=True)

for i, seq_id in enumerate(SCENARIOS):
    config = TEMPLATE.copy()
    demo_name = f"demo_new_{i+1}"
    
    config["sim_name"] = demo_name
    config["seq_id"] = seq_id
    config["output_dir"] = f"output/{demo_name}/"
    
    filename = f"configs/{demo_name}.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Created {filename}")
