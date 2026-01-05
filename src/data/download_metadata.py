import urllib.request
import zipfile
from pathlib import Path


def download_metadata():
    base_url = "https://os.unil.cloud.switch.ch/fma/"
    metadata_url = f"{base_url}fma_metadata.zip"

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata_zip = data_dir / "fma_metadata.zip"

    if not metadata_zip.exists():
        print(f"Downloading metadata from {metadata_url}...")
        urllib.request.urlretrieve(metadata_url, metadata_zip)
        print(f"Downloaded to {metadata_zip}")
    else:
        print(f"Metadata already exists at {metadata_zip}")

    extract_dir = data_dir / "fma_metadata"
    if not extract_dir.exists():
        print(f"Extracting metadata...")
        with zipfile.ZipFile(metadata_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted to {extract_dir}")
    else:
        print(f"Metadata already extracted at {extract_dir}")

    return extract_dir


def print_download_instructions():
    print("\n" + "="*60)
    print("MANUAL STEP REQUIRED")
    print("="*60)
    print("\nFMA Medium dataset (25GB) must be downloaded manually:")
    print("\n1. Visit: https://os.unil.cloud.switch.ch/fma/fma_medium.zip")
    print("2. Download fma_medium.zip (25 GB)")
    print("3. Extract to: data/raw/fma_medium/")
    print("\nExpected structure:")
    print("data/raw/fma_medium/")
    print("  - 000/")
    print("  - 001/")
    print("  - ...")
    print("\nAfter extraction, run: python src/data/verify_dataset.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    metadata_path = download_metadata()
    print(f"\nMetadata ready at: {metadata_path}")
    print_download_instructions()
