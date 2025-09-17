"""
Script to download Fashion Product Images dataset from Kaggle
"""
import os
import zipfile
import shutil
from pathlib import Path
import subprocess
import sys
import requests
from tqdm import tqdm
import config

def check_kaggle_credentials():
    """
    Check if Kaggle credentials are set up
    """
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if not kaggle_json.exists():
        print("❌ Kaggle credentials not found!")
        print("\nTo download the dataset, you need to:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to Account Settings -> API -> Create New API Token")
        print("3. Save the downloaded kaggle.json to ~/.kaggle/")
        print("4. Run this script again")
        return False

    # Set permissions for kaggle.json
    os.chmod(kaggle_json, 0o600)
    return True

def download_with_kaggle_api():
    """
    Download dataset using Kaggle API
    """
    try:
        import kaggle

        print(f"📥 Downloading dataset: {config.KAGGLE_DATASET}")

        # Create data directory
        config.DATA_DIR.mkdir(exist_ok=True)

        # Download dataset
        kaggle.api.dataset_download_files(
            config.KAGGLE_DATASET,
            path=config.DATA_DIR,
            unzip=True
        )

        print("✅ Dataset downloaded successfully!")
        return True

    except ImportError:
        print("❌ Kaggle package not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return download_with_kaggle_api()
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False


def verify_dataset():
    """
    Verify that the dataset is properly downloaded
    """

    if not config.STYLES_CSV.exists():
        print("❌ styles.csv not found")
        return False

    if not config.IMAGES_DIR.exists():
        print("❌ Images directory not found")
        return False

    # Count images
    image_files = list(config.IMAGES_DIR.glob("*.jpg")) + list(config.IMAGES_DIR.glob("*.png"))

    print(f"\n📊 Dataset Statistics:")
    print(f"   - Metadata file: {config.STYLES_CSV}")
    print(f"   - Number of images: {len(image_files)}")

    if len(image_files) == 0:
        print("⚠️  No images found in the dataset")
        return False

    return True

def main():
    """
    Main function to download and set up the dataset
    """
    print("="*50)
    print("Fashion Product Images Dataset Downloader")
    print("="*50)

    # Check if dataset already exists
    if verify_dataset():
        print("\n✅ Dataset already exists and is verified!")
        response = input("Do you want to re-download? (y/n): ").lower()
        if response != 'y':
            return

    print("\nChoose download method:")
    print("1. Download full dataset from Kaggle (requires Kaggle account)")
    print("2. Create sample dataset for testing (no account needed)")

    choice = input("\nEnter your choice (1): ").strip()

    if choice == '1':
        if download_with_kaggle_api():
            verify_dataset()
        else:
            print("\n⚠️  Failed to download from Kaggle. Try Again...")

    # Final verification
    if verify_dataset():
        print("\n✅ Dataset is ready to use!")
        print("Run 'streamlit run app.py' to start the Fashion Shopping Assistant")
    else:
        print("\n❌ Dataset setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
