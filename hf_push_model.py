from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="models/category_model",
    repo_id="DhanushGWU1995/ecris-category-model",
    repo_type="model"
)

print("Model uploaded successfully")
