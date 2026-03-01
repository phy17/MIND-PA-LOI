import os
import json
import sys

def find_and_fix_config():
    home_dir = os.path.expanduser("~")
    target_filename = "antigravity-accounts.json"
    target_project_id = "adroit-producer-487207-r4"
    
    # 默认修复位置 (如果找不到文件，将在此处创建)
    default_fix_path = os.path.join(home_dir, ".config", "opencode", target_filename)
    
    print(f"🔍 正在从 {home_dir} 全盘搜索 {target_filename}...")
    
    found_path = None
    
    # 1. 尝试常见路径
    likely_paths = [
        default_fix_path,
        os.path.join(home_dir, "Library", "Application Support", "Google", "Antigravity", target_filename),
        os.path.join(home_dir, "Library", "Application Support", "antigravity", target_filename),
        os.path.join(home_dir, ".antigravity", target_filename),
        os.path.join(home_dir, ".gemini", target_filename),
    ]
    
    for p in likely_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    # 2. 全盘搜索 (如果常见路径没有)
    if not found_path:
        # print("   (深度搜索中，请稍候...)")
        for root, dirs, files in os.walk(home_dir, topdown=True):
            # 排除由于权限或无关紧要的目录
            if ".Trash" in dirs: dirs.remove(".Trash")
            if "Downloads" in dirs: dirs.remove("Downloads")
            if "node_modules" in dirs: dirs.remove("node_modules")
            if ".git" in dirs: dirs.remove(".git")
            if "Library" in root and "Caches" in dirs: dirs.remove("Caches")
            
            if target_filename in files:
                found_path = os.path.join(root, target_filename)
                break
    
    # 3. 如果还是找不到 -> 在默认位置创建
    if not found_path:
        print(f"❌ 全盘搜索未找到 {target_filename}。")
        print(f"⚠️ 这是一个 Ghost Login 现象: 配置文件丢失，但 IDE 认为已登录。")
        print(f"🛠 正在尝试创建修复文件至: {default_fix_path}")
        
        found_path = default_fix_path
        # 确保目录存在
        os.makedirs(os.path.dirname(found_path), exist_ok=True)
        # 创建新文件内容 (假设只有一个账号)
        data = {
            "accounts": [
                {
                    "projectId": target_project_id,
                    "type": "USER_ACCOUNT" # 猜测类型，通常不影响
                }
            ]
        }
    else:
        print(f"✅ 找到文件: {found_path}")
        try:
            with open(found_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content) if content.strip() else {}
        except Exception as e:
            print(f"❌ 读取错误: {e}")
            data = {}

    # 4. 写入/更新 Project ID
    try:
        modified = False
        
        def inject(d):
            if isinstance(d, dict):
                d["projectId"] = target_project_id
                return True
            return False

        # 处理数据结构
        if isinstance(data, list):
            for item in data:
                if inject(item): modified = True
        elif isinstance(data, dict):
            if "accounts" in data and isinstance(data["accounts"], list):
                if not data["accounts"]: # 空列表
                     data["accounts"].append({})
                for acc in data["accounts"]:
                    if inject(acc): modified = True
            else:
                # 可能是空字典或根对象
                if not data: 
                    # 初始化结构
                    data = {"accounts": [{"projectId": target_project_id}]}
                    modified = True
                else:
                    if inject(data): modified = True

        # 始终写入（如果是新创建的文件或有修改）
        with open(found_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print("⚡️ 成功写入/更新 Project ID!")
        print("-" * 40)
        print(f"文件路径: {found_path}")
        print("内容预览:")
        print(json.dumps(data, indent=2))
        print("-" * 40)
        print("💡 请重启 Antigravity IDE 以生效。")

    except Exception as e:
        print(f"❌ 写入文件时发生错误: {e}")

if __name__ == "__main__":
    find_and_fix_config()
