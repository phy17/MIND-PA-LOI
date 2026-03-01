import os
import json

def fix_config():
    # 1. Target Path
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, ".config", "opencode")
    config_file = os.path.join(config_dir, "antigravity-accounts.json")
    
    print(f"Target file: {config_file}")

    # 2. Ensure Directory
    if not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir)
            print(f"Created directory: {config_dir}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            return
    else:
        print(f"Directory exists: {config_dir}")

    # 3. Force Write Payload
    payload = {
        "accounts": [
            {
                "email": "manual_fix@bypass.com",
                "type": "google",
                "projectId": "bamboo-precept-lgxtn",
                "valid": True
            }
        ]
    }

    try:
        with open(config_file, "w") as f:
            json.dump(payload, f, indent=2)
        print("Successfully wrote configuration payload.")
    except IOError as e:
        print(f"Error writing to file: {e}")
        return

    # 4. Verification
    try:
        with open(config_file, "r") as f:
            data = json.load(f)
            # Safely access the nested structure
            accounts = data.get("accounts", [])
            if accounts and isinstance(accounts, list):
                project_id = accounts[0].get("projectId")
                print(f"Verification - Project ID in file: {project_id}")
                
                if project_id == "bamboo-precept-lgxtn":
                     print("SUCCESS: Project ID matches expected value.")
                else:
                     print(f"FAILURE: Project ID mismatch. Got: {project_id}")
            else:
                print("FAILURE: Verification failed, 'accounts' list missing or empty.")

    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading back file for verification: {e}")

if __name__ == "__main__":
    fix_config()
