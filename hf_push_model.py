from huggingface_hub import HfApi
import os

api = HfApi()

repo_id = "DhanushGWU1995/ecris-category-model"
repo_type = "model"

print("🗑️  Deleting existing files from repository...")

# Get list of all files in the repository
try:
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    print(f"   Found {len(repo_files)} existing files")
    
    # Delete each file
    for file_path in repo_files:
        # Skip .gitattributes as it's automatically created by HF
        if file_path != ".gitattributes":
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=f"Remove old file: {file_path}"
                )
                print(f"   ✓ Deleted: {file_path}")
            except Exception as e:
                print(f"   ⚠️  Could not delete {file_path}: {e}")
    
    print("✓ Repository cleaned successfully!")
    
except Exception as e:
    print(f"⚠️  Note: Could not list/delete existing files: {e}")
    print("   Proceeding with upload (files will be overwritten)...")

print("\n📤 Uploading new model files...")

api.upload_folder(
    folder_path="models",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload fresh model artifacts",
    delete_patterns=["*"]  # This also helps ensure clean upload
)

print("✓ Models folder uploaded!")

# Upload lstm_config.json from reports folder (needed for app)
print("\n📊 Uploading LSTM config...")
try:
    if os.path.exists("reports/lstm_config.json"):
        api.upload_file(
            path_or_fileobj="reports/lstm_config.json",
            path_in_repo="lstm_config.json",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload LSTM configuration"
        )
        print("✓ LSTM config uploaded!")
    else:
        print("⚠️  lstm_config.json not found in reports/")
except Exception as e:
    print(f"⚠️  Could not upload config: {e}")

print("\n✅ All files uploaded successfully!")
