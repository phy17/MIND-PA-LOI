import os

def delete_config():
    file_path = os.path.expanduser("~/.config/opencode/antigravity-accounts.json")
    print(f"Target file: {file_path}")

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print("-" * 30)
            print(f"File deleted. Please re-add your account in the Antigravity Tool now.")
            print("-" * 30)
        except OSError as e:
            print(f"Error removing file: {e}")
    else:
        print("File not found.")

if __name__ == "__main__":
    delete_config()
