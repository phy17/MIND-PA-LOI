import os
import shutil
import time

def deep_clean_antigravity():
    home = os.path.expanduser("~")
    # 确认 App 路径
    antigravity_data_dir = os.path.join(home, "Library", "Application Support", "Antigravity")
    
    # 1. 杀掉进程 (确保彻底关闭)
    # pkill 不一定能杀掉所有 helper，所以要多杀几次
    print("💀 正在强制终止所有 Antigravity 进程...")
    os.system("pkill -9 -f Antigravity")
    time.sleep(2) # 等待释放文件锁

    if not os.path.exists(antigravity_data_dir):
        print("❌ 未找到数据目录: " + antigravity_data_dir)
        return

    # 2. 清理认证与缓存 (保留关键数据，只删除认证相关的缓存)
    # 这些目录是导致 Auth Loop 和 Profile Picture 错误的罪魁祸首
    targets_to_remove = [
        "auth-tokens",   # 关键：认证令牌缓存
        "Cache",         # 缓存文件
        "GPUCache",      # GPU缓存
        "Code Cache",    # 代码缓存
        "CachedData",    # 扩展数据缓存
        "Cookies",       # Cookie
        "Cookies-journal",
        "Local Storage", # 本地存储
        "Session Storage", # 会话存储
        "Network Persistent State", # 网络状态
    ]

    print("\n🧹 开始深度清理缓存...")
    for target in targets_to_remove:
        full_path = os.path.join(antigravity_data_dir, target)
        if os.path.exists(full_path):
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                print(f"✅ 已删除: {target}")
            except Exception as e:
                print(f"⚠️ 无法删除 {target}: {e}")
        else:
            # print(f"   (未找到 {target}，跳过)")
            pass

    # 3. 强制写入正确的 accounts 配置文件
    # 这一步非常重要，必须在清除缓存后立即写入，防止它再次生成错误的空文件
    accounts_file = os.path.join(antigravity_data_dir, "antigravity-accounts.json")
    correct_content = '''{
  "accounts": [
    {
      "projectId": "adroit-producer-487207-r4",
      "type": "USER_ACCOUNT"
    }
  ]
}'''
    
    try:
        with open(accounts_file, 'w', encoding='utf-8') as f:
            f.write(correct_content)
        print(f"\n✅ 已重建标准的 {accounts_file}")
        print("   -> Project ID: adroit-producer-487207-r4")
    except Exception as e:
        print(f"❌ 配置文件写入失败: {e}")

    # 4. 同时更新 .config 下的文件，以防万一
    config_dir = os.path.join(home, ".config", "opencode")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    config_file_2 = os.path.join(config_dir, "antigravity-accounts.json")
    try:
        with open(config_file_2, 'w', encoding='utf-8') as f:
            f.write(correct_content)
        print(f"✅ 已同步更新 {config_file_2}")
    except:
        pass

    print("\n✨ 清理完成！")
    print("🚀 请从 Dock 或 Launchpad 重新启动 Antigravity。它将像新安装一样启动。")

if __name__ == "__main__":
    deep_clean_antigravity()
