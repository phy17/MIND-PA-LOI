import os
import json

def search_files():
    home = os.path.expanduser("~")
    patterns = ["antigravity-accounts.json", "google_accounts.json", "accounts.json", "application_default_credentials.json"]
    search_roots = [
        os.path.join(home, ".config"),
        os.path.join(home, "Library", "Application Support"),
        os.path.join(home, ".gemini"),
        os.path.join(home, ".antigravity"),
        os.path.join(home, ".oam"), # Guess
        os.path.join(home, ".open-agent-manager") # Guess
    ]

    found_files = []

    print("🔍 Searching for auth config files...")
    for root_dir in search_roots:
        if not os.path.exists(root_dir):
            continue
        
        for root, dirs, files in os.walk(root_dir):
            # Optimization: Skip cache and trash dirs
            if "Caches" in dirs: dirs.remove("Caches")
            if "Cache" in dirs: dirs.remove("Cache")
            if "node_modules" in dirs: dirs.remove("node_modules")
            
            for file in files:
                if file in patterns or (file.endswith(".json") and "account" in file):
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    print(f"📄 Found: {full_path}")
    
    return found_files

def check_env():
    print("\n🔍 Checking Environment Variables...")
    keys = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS", "ANTIGRAVITY_PROJECT_ID"]
    for k in keys:
        val = os.environ.get(k)
        print(f"  {k}: {val if val else '(not set)'}")

if __name__ == "__main__":
    files = search_files()
    check_env()
