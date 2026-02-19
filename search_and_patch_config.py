import os
import json
import sys

def search_and_fix():
    home_dir = os.path.expanduser("~")
    target_filename = "antigravity-accounts.json"
    exclude_dirs = {"Caches", ".Trash", "Downloads", "node_modules", ".git"}
    
    print(f"🔍 Searching for {target_filename} starting from {home_dir}...")
    
    found_files = []
    
    for root, dirs, files in os.walk(home_dir, topdown=True):
        # Optimizations: Prune search tree
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        if "Library" in root and "Caches" in dirs:
             dirs.remove("Caches")
             
        if target_filename in files:
            full_path = os.path.join(root, target_filename)
            found_files.append(full_path)
            # We don't break here to find ALL instances if multiple exist
            
    if not found_files:
        print("❌ File definitively not found. The tool has not saved it yet.")
        return

    # Fix ALL found instances
    for file_path in found_files:
        print(f"FOUND config file at: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip(): 
                    print(f"⚠️ File is empty: {file_path}")
                    continue
                data = json.loads(content)
                
            accounts = data.get("accounts", [])
            if not isinstance(accounts, list):
                print(f"⚠️ 'accounts' is not a list in {file_path}")
                continue

            target_email = "ddd756042@gmail.com"
            target_idx = -1
            
            # Locate
            for i, acc in enumerate(accounts):
                # Loose matching: check if target_email is contained or exact match
                if target_email in acc.get("email", ""):
                    target_idx = i
                    break
            
            if target_idx != -1:
                print(f"✅ Found account '{target_email}' at index {target_idx}")
                
                # Extract & Patch
                acc = accounts.pop(target_idx)
                acc["projectId"] = "bamboo-precept-lgxtn"
                acc["valid"] = True
                
                # Reorder to top
                accounts.insert(0, acc)
                data["accounts"] = accounts
                
                # Save
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"🎉 Successfully patched and saved: {file_path}")
                
            else:
                print(f"⚠️ Account '{target_email}' NOT FOUND in {file_path}")
                emails = [a.get("email") for a in accounts]
                print(f"   Existing accounts: {emails}")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

if __name__ == "__main__":
    search_and_fix()
