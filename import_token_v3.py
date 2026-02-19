import os
import json
import shutil

def import_and_fix_token_v3():
    # 1. 源文件 (raw_accounts.json)
    # 优先查找 Desktop，其次查找主目录
    home = os.path.expanduser("~")
    possible_sources = [
        os.path.join(home, "Desktop", "raw_accounts.json"),
        os.path.join(home, "raw_accounts.json")
    ]
    
    source_file = None
    for p in possible_sources:
        if os.path.exists(p):
            source_file = p
            break
            
    if not source_file:
        print("❌ 错误: 未能在桌面或主目录找到 'raw_accounts.json'")
        return

    # 目标文件 (Antigravity 实际读取的位置)
    target_dir = os.path.expanduser("~/Library/Application Support/Antigravity")
    target_file = os.path.join(target_dir, "antigravity-accounts.json")

    print(f"📖 读取源文件: {source_file}")

    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                print("⚠️ 源文件内容为空")
                return
            raw_data = json.loads(content)

        # 2. 智能解析 JSON 结构
        accounts = []
        
        # 情况 A: 根对象就是列表 [{}, {}]
        if isinstance(raw_data, list):
            accounts = raw_data
        # 情况 B: 根对象包含 "accounts" 键 {"accounts": [...]}
        elif isinstance(raw_data, dict):
            if "accounts" in raw_data and isinstance(raw_data["accounts"], list):
                accounts = raw_data["accounts"]
            # 情况 C: 根对象本身就是一个账号信息 (包含 accessToken)
            elif "accessToken" in raw_data or "email" in raw_data:
                accounts = [raw_data]
        
        if not accounts:
            print("❌ 错误: 未能在源文件中提取到有效的账号列表")
            # 打印调试信息
            print(f"   数据结构类型: {type(raw_data)}")
            if isinstance(raw_data, dict):
                print(f"   键: {raw_data.keys()}")
            return

        # 3. 筛选并修复目标账号
        target_account = None
        target_email = "ddd756042@gmail.com"
        
        # 尝试精确匹配
        for acc in accounts:
            if acc.get("email") == target_email:
                target_account = acc
                break
        
        # 如果没找到，兜底策略：使用列表中的第一个账号
        if not target_account and len(accounts) > 0:
            print(f"⚠️ 未找到精确匹配 '{target_email}' 的账号")
            print(f"   将使用列表中的第 1 个账号作为替代: {accounts[0].get('email', 'Unknown')}")
            target_account = accounts[0]
            
        if not target_account:
            print("❌ 错误: 无法确定目标账号")
            return

        print(f"✅ 锁定账号: {target_account.get('email', 'Unknown')}")

        # === 关键注入 ===
        target_account["projectId"] = "bamboo-precept-lgxtn"
        target_account["valid"] = True
        
        # 4. 构造最终配置并写入
        final_payload = {
            "accounts": [target_account]
        }
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(final_payload, f, indent=2)
            
        print("-" * 40)
        print(f"🎉 成功！配置已修复并写入: {target_file}")
        print(f"🔑 Project ID: {target_account.get('projectId')}")
        print("-" * 40)
        print("🚀 请务必重启 Antigravity IDE 以生效。")

    except Exception as e:
        print(f"❌ 发生未预期的错误: {e}")

if __name__ == "__main__":
    import_and_fix_token_v3()
