import os
import json
import sys

def surgical_fix():
    # 1. Target Path
    file_path = os.path.expanduser("~/.config/opencode/antigravity-accounts.json")
    print(f"Target file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 2. Load Configuration
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
    target_account = None
    target_index = -1

    # 3. Locate Target
    for i, account in enumerate(accounts):
        if account.get("email") == target_email:
            target_account = account
            target_index = i
            break

    if target_account is None:
        print(f"Error: Account with email '{target_email}' not found in configuration.")
        # Optional: Print available emails to help debugging
        emails = [acc.get("email") for acc in accounts]
        print(f"Available accounts: {emails}")
        return

    print(f"Found account '{target_email}' at index {target_index}.")

    # 4. Patch Data
    print("Patching account data...")
    target_account["projectId"] = "bamboo-precept-lgxtn"
    target_account["valid"] = True
    # We leave other fields intact as requested

    # 5. Reorder (Crucial)
    # Remove from current position
    accounts.pop(target_index)
    # Insert at top
    accounts.insert(0, target_account)
    data["accounts"] = accounts
    print("Account moved to index 0 (Default).")

    # 6. Save & Verify
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print("-" * 30)
        print("SUCCESS: Configuration updated successfully.")
        print(f"Active Account: {data['accounts'][0]['email']}")
        print(f"Project ID: {data['accounts'][0]['projectId']}")
        print(f"Valid Status: {data['accounts'][0]['valid']}")
        print("-" * 30)

    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    surgical_fix()
