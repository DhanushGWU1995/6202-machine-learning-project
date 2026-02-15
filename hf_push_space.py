"""
Upload app files to Hugging Face Spaces
Uploads app_lstm.py and requirement.txt to ECRIS space
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

# Note: We don't upload models/reports folders to Space
# The app will download models from DhanushGWU1995/ecris-category-model
print("\n� Note: Models will be loaded from Hugging Face model repository")
print("   Repository: DhanushGWU1995/ecris-category-model")

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
