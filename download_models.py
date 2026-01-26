#!/usr/bin/env python3
"""
Script to download .pt model files from Hugging Face repository
specifically for the Indian ID Validator project.
"""

import os
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path


def download_pt_models(repo_id="logasanjeev/indian-id-validator", local_dir="./models"):
    """
    Downloads only .pt model files from the specified Hugging Face repository.
    
    Args:
        repo_id (str): The Hugging Face repository ID
        local_dir (str): Local directory to save the models
    """
    print(f"Connecting to Hugging Face repository: {repo_id}")
    
    # Create models directory if it doesn't exist
    models_dir = Path(local_dir)
    models_dir.mkdir(exist_ok=True)
    
    # List all files in the repository
    print("Fetching file list from repository...")
    all_files = list_repo_files(repo_id=repo_id)
    
    # Filter for .pt files only
    pt_files = [f for f in all_files if f.lower().endswith('.pt')]
    
    if not pt_files:
        print("No .pt files found in the repository.")
        return
    
    print(f"Found {len(pt_files)} .pt model files:")
    for file in pt_files:
        print(f"  - {file}")
    
    # Download each .pt file
    for file in pt_files:
        print(f"\nDownloading {file}...")
        try:
            # Download the file to a temporary location
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                # local_dir=models_dir,  # Don't use local_dir to get the file in cache first
                local_dir_use_symlinks=False
            )

            # Copy the downloaded file to our models directory
            import shutil
            filename_only = os.path.basename(file)
            destination_path = models_dir / filename_only
            shutil.copy2(downloaded_path, destination_path)

            print(f"✓ Successfully downloaded {filename_only} to {destination_path}")
        except Exception as e:
            print(f"✗ Failed to download {file}: {str(e)}")
    
    print(f"\nCompleted! All .pt models have been saved to '{local_dir}'")


if __name__ == "__main__":
    # Default values
    REPO_ID = "logasanjeev/indian-id-validator"
    LOCAL_DIR = "./models"
    
    # Allow override via environment variables
    repo_id = os.environ.get("HF_REPO_ID", REPO_ID)
    local_dir = os.environ.get("MODELS_DIR", LOCAL_DIR)
    
    download_pt_models(repo_id=repo_id, local_dir=local_dir)