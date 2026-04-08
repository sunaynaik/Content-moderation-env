from huggingface_hub import HfApi
import os

print("\n🚀 Starting upload to Hugging Face...")
token = os.environ.get("HF_TOKEN")
if not token:
    token = input("Please paste your HF Write Token (hf_...): ").strip()

api = HfApi()

try:
    print("Uploading files... this might take a few seconds.")
    api.upload_folder(
        folder_path=".",
        path_in_repo=".",
        repo_id="sunaynaik345/content-moderation-env",
        repo_type="space",
        token=token,
        ignore_patterns=["upload.py", "__pycache__/*", "*.pyc", ".git/*"]
    )
    print("✅ Uploaded successfully! Go refresh your browser.")
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
