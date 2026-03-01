import os
import json

def inspect_gemini_config():
    # Target file
    target = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")
    
    if not os.path.exists(target):
        print(f"❌ File not found: {target}")
        return

    size = os.path.getsize(target)
    print(f"📄 File: {target} (Size: {size} bytes)")
    
    try:
        with open(target, "r") as f:
            content = f.read()
            print("--- Content Start ---")
            print(content)
            print("--- Content End ---")
            
            # Try parsing JSON
            try:
                data = json.loads(content)
                print("✅ Valid JSON")
                # Check for projectId
                if "projectId" in data:
                    print(f"🔑 Found projectId: '{data['projectId']}'")
                elif "google_cloud_project" in data:
                    print(f"🔑 Found google_cloud_project: '{data['google_cloud_project']}'")
                else:
                    print("⚠️ No projectId found in top level keys.")
                    
            except json.JSONDecodeError:
                print("❌ Invalid JSON content")
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    inspect_gemini_config()
