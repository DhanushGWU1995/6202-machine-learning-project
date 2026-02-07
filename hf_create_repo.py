from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="DhanushGWU1995/ecris-category-model",
    repo_type="model",
    exist_ok=True
)

print("Repository created or already exists")
