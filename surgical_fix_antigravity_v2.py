import os
import json
import sys

def surgical_fix_v2():
    # 1. Target Path
    file_path = os.path.expanduser("~/.config/opencode/antigravity-accounts.json")
    print(f"Target file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 2. Read Configuration
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    accounts = data.get("accounts", [])
    if not isinstance(accounts, list):
        print("Error: 'accounts' field is not a list.")
        return

    target_email = "ddd756042@gmail.com"
    target_idx = -1
    
    # 3. Locate Target
    for i, acc in enumerate(accounts):
        if acc.get("email") == target_email:
            target_idx = i
            break
            
    if target_idx == -1:
        print(f"Error: Account '{target_email}' STILL NOT FOUND.")
        print("Please refresh the tool again.")
        # Debug helper: List present emails
        emails_present = [acc.get("email") for acc in accounts]
        print(f"Current accounts in file: {emails_present}")
        return

    # 4. Patch & Reorder
    print(f"Found account '{target_email}' at index {target_idx}.")
    
    # Extract the account
    account_to_fix = accounts.pop(target_idx)
    
    # Update fields
    account_to_fix["projectId"] = "bamboo-precept-lgxtn" # The Fix
    account_to_fix["valid"] = True
    
    # Insert at top (Index 0)
    accounts.insert(0, account_to_fix)
    
    # Update data structure
    data["accounts"] = accounts
    
    # 5. Save
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print("-" * 40)
        print(f"Success! Account {target_email} fixed and set as default.")
        print(f"Current Default: {data['accounts'][0]['email']}")
        print(f"Project ID: {data['accounts'][0]['projectId']}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    surgical_fix_v2()
