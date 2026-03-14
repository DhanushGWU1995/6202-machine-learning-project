"""hf_push_space.py

Upload app files (and required artifacts) to a Hugging Face Space.

This script uploads:
- `app_lstm.py` -> `src/streamlit_app.py`
- `requirement.txt` -> `requirements.txt`
- `models/` folder -> `models/`
- `reports/` folder -> `reports/`
"""

from huggingface_hub import HfApi
import os

api = HfApi()

space_id = "DhanushGWU1995/ECRIS"
space_type = "space"

print("🚀 UPLOADING TO HUGGING FACE SPACE")
print("=" * 70)
print(f"📍 Space: {space_id}")
print("=" * 70)

# Delete existing files first (clean slate)
print("\n🗑️  Cleaning old app file from space...")
try:
    # Only delete the specific files we're replacing
    files_to_delete = ["src/streamlit_app.py", "requirements.txt"]
    
    for file_path in files_to_delete:
        try:
            api.delete_file(
                path_in_repo=file_path,
                repo_id=space_id,
                repo_type=space_type,
                commit_message=f"Remove old file: {file_path}"
            )
            print(f"   ✓ Deleted: {file_path}")
        except Exception as e:
            print(f"   ℹ️  {file_path} not found (will be created)")
    
    print("✓ Cleanup completed!")
except Exception as e:
    print(f"⚠️  Note: {e}")
    print("   Proceeding with upload...")


print("\n🧹 Cleaning old artifact folders from space (models/, reports/)...")
try:
    repo_files = api.list_repo_files(repo_id=space_id, repo_type=space_type)
    targets = ("models/", "reports/")
    to_delete = [p for p in repo_files if p.startswith(targets)]

    if not to_delete:
        print("   ℹ️  No existing models/ or reports/ files found")
    else:
        print(f"   Found {len(to_delete)} files to delete")
        for file_path in to_delete:
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=space_id,
                    repo_type=space_type,
                    commit_message=f"Remove old artifact: {file_path}",
                )
                print(f"   ✓ Deleted: {file_path}")
            except Exception as e:
                print(f"   ⚠️  Could not delete {file_path}: {e}")
                errors.append(f"delete {file_path}: {e}")

    print("✓ Artifact cleanup completed!")
except Exception as e:
    print(f"⚠️  Note: Could not list/delete artifact files: {e}")
    print("   Proceeding with upload (files may be overwritten)...")

print("\n📤 Uploading files to space...")

# List of files to upload
files_to_upload = [
    ("app_lstm.py", "src/streamlit_app.py"),  # Upload to src/streamlit_app.py
    ("requirement.txt", "requirements.txt")  # Rename to requirements.txt (standard)
]

upload_count = 0
errors = []

for local_file, remote_file in files_to_upload:
    if not os.path.exists(local_file):
        print(f"   ⚠️  File not found: {local_file}")
        errors.append(f"{local_file} not found")
        continue
    
    try:
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=remote_file,
            repo_id=space_id,
            repo_type=space_type,
            commit_message=f"Upload {remote_file}"
        )
        
        file_size = os.path.getsize(local_file)
        print(f"   ✓ Uploaded: {local_file} → {remote_file} ({file_size:,} bytes)")
        upload_count += 1
        
    except Exception as e:
        print(f"   ❌ Error uploading {local_file}: {e}")
        errors.append(f"{local_file}: {e}")

def upload_folder_if_present(local_dir: str, remote_dir: str, *, ignore_patterns: list[str] | None = None) -> None:
    """Upload a local folder to the Space repo if it exists.

    We keep this defensive because Spaces repos have size limits; ignore patterns
    let us skip caches and other non-essential files.
    """
    if not os.path.isdir(local_dir):
        print(f"   ℹ️  Folder not found, skipping: {local_dir}/")
        return

    try:
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=remote_dir,
            repo_id=space_id,
            repo_type=space_type,
            commit_message=f"Upload folder: {remote_dir}/",
            ignore_patterns=ignore_patterns or [],
        )
        print(f"   ✓ Uploaded folder: {local_dir}/ → {remote_dir}/")
    except Exception as e:
        print(f"   ❌ Error uploading folder {local_dir}/: {e}")
        errors.append(f"{local_dir}/: {e}")


print("\n📦 Uploading artifact folders (models/, reports/)...")

# Keep ignores conservative and focused on common junk.
# NOTE: If your model files are very large (>~10s/100s of MB), consider Git LFS
# or hosting them in a model repo and downloading at runtime.
common_ignores = [
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/.ipynb_checkpoints/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/.venv/**",
    "**/venv/**",
]

upload_folder_if_present("models", "models", ignore_patterns=common_ignores)
upload_folder_if_present("reports", "reports", ignore_patterns=common_ignores)

# Summary
print("\n" + "=" * 70)
if errors:
    print("⚠️  UPLOAD COMPLETED WITH WARNINGS")
    print("=" * 70)
    print(f"✓ Successfully uploaded: {upload_count} items")
    print(f"⚠️  Errors/Warnings: {len(errors)}")
    for error in errors:
        print(f"   - {error}")
else:
    print("✅ ALL FILES UPLOADED SUCCESSFULLY!")
    print("=" * 70)
    print(f"✓ Total items uploaded: {upload_count}")

print(f"\n🔗 Your space is available at:")
print(f"   https://huggingface.co/spaces/{space_id}")
print("\n⏱️  Note: Space may take 1-2 minutes to rebuild and start")
print("=" * 70)
