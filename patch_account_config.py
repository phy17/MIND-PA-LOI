import os
import json
import sys

def patch_account_id():
    file_path = os.path.expanduser("~/.config/opencode/antigravity-accounts.json")
    print(f"Target file: {file_path}")

    if not os.path.exists(file_path):
        print("Error: File not found at target path.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return

    accounts = data.get("accounts", [])
    if not isinstance(accounts, list):
        print("Error: 'accounts' key is not a list in the JSON.")
        return

    target_email = "ddd756042@gmail.com"
    target_idx = -1

    # Locate the account
    for i, acc in enumerate(accounts):
       if acc.get("email") == target_email:
           target_idx = i
           break
    
    if target_idx == -1:
        print(f"Error: Account '{target_email}' not found. Please check the tool.")
        emails_present = [acc.get("email") for acc in accounts]
        print(f"Emails found in file: {emails_present}")
        return

    print(f"Account '{target_email}' found at index {target_idx}.")

    # Patch the Account
    account_to_fix = accounts.pop(target_idx)
    account_to_fix["projectId"] = "bamboo-precept-lgxtn"
    account_to_fix["valid"] = True
    
    # Reorder (Move to top)
    accounts.insert(0, account_to_fix)
    data["accounts"] = accounts

    # Write Back
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print("-" * 30)
        print(f"Success! Account patched with Project ID: bamboo-precept-lgxtn")
        print(f"Active Account set to: {data['accounts'][0]['email']}")
        print(f"Valid Status: {data['accounts'][0]['valid']}")
        print("-" * 30)

    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    patch_account_id()
