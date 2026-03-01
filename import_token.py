import os
import json
import shutil

def import_and_fix_token():
    # 1. 查找源文件 (raw_accounts.json)
    # 不仅仅在 Desktop，可能也在用户根目录，我们都找一下
    home = os.path.expanduser("~")
    search_paths = [
        os.path.join(home, "Desktop", "raw_accounts.json"),
        os.path.join(home, "raw_accounts.json")
    ]
    
    source_file = None
    for p in search_paths:
        if os.path.exists(p):
            source_file = p
            break
            
    if not source_file:
        print("❌ 错误: 未能在桌面或主目录找到 raw_accounts.json")
        return

    print(f"📖 读取源文件: {source_file}")

    # 2. 读取并处理数据
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                print("⚠️ 源文件是空的")
                return
            source_data = json.loads(content)
            
        accounts = source_data.get("accounts", [])
        if not accounts:
            # 也许根对象本身就是账号字典？
            if "accessToken" in source_data:
                accounts = [source_data]
            else:
                print("⚠️ 源文件中没有找到有效的账户数据结构")
                return

        # 找到目标账号 (优先匹配 ddd756042)
        target_account = None
        for acc in accounts:
            if acc.get("email") == "ddd756042@gmail.com":
                target_account = acc
                break
        
        if not target_account:
            print("⚠️ 源文件中没找到 ddd756042@gmail.com，将使用第一个可用账号。")
            target_account = accounts[0]

        email = target_account.get('email', 'Unknown')
        print(f"✅ 提取到账号: {email}")

        # === 关键注入 ===
        target_account["projectId"] = "bamboo-precept-lgxtn"
        target_account["valid"] = True
        
        # 构造最终数据
        final_payload = {
            "accounts": [target_account]
        }
        
        # 3. 写入目标位置 (确保写入 Antigravity 实际读取的位置)
        # 注意：这里我们写入 Application Support 路径，这是你系统上 Antigravity 真正读取的位置
        target_dir = os.path.expanduser("~/Library/Application Support/Antigravity")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        target_file = os.path.join(target_dir, "antigravity-accounts.json")

        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(final_payload, f, indent=2)
            
        print("-" * 40)
        print(f"🎉 成功导入并修复配置！")
        print(f"📂 目标路径: {target_file}")
        print(f"🔑 Project ID 已设置为: {target_account['projectId']}")
        print("-" * 40)
        print("🚀 现在，请重启 Antigravity IDE。")

    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")

if __name__ == "__main__":
    import_and_fix_token()
