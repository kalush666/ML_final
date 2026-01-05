import os
import subprocess
from pathlib import Path


def download_via_kaggle():
    print("Attempting Kaggle download (fastest option)...")
    print("\nSetup required:")
    print("1. Go to: https://www.kaggle.com/")
    print("2. Sign in or create account")
    print("3. Go to: https://www.kaggle.com/settings/account")
    print("4. Scroll to 'API' section")
    print("5. Click 'Create New Token'")
    print("6. Save kaggle.json to: C:\\Users\\jonat\\.kaggle\\kaggle.json")
    print("\nAfter setup, run:")
    print("pip install kaggle")
    print("kaggle datasets download -d undisclosed/fma-medium")


def download_via_aria2c():
    print("Attempting aria2c download (multi-connection, faster)...")

    url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    aria2_check = subprocess.run(
        ["aria2c", "--version"],
        capture_output=True,
        text=True
    )

    if aria2_check.returncode != 0:
        print("\naria2c not installed!")
        print("Install with: winget install aria2.aria2")
        print("or download from: https://github.com/aria2/aria2/releases")
        return False

    print(f"\nDownloading to {output_dir}/fma_medium.zip")
    print("Using 16 connections for faster download...")

    cmd = [
        "aria2c",
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        "--file-allocation=none",
        "--dir", str(output_dir),
        "--out", "fma_medium.zip",
        url
    ]

    subprocess.run(cmd)
    return True


def download_via_curl():
    print("Attempting curl download (resume support)...")

    url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fma_medium.zip"

    print(f"\nDownloading to {output_file}")
    print("Can be resumed if interrupted (Ctrl+C to pause)")

    cmd = [
        "curl",
        "-C", "-",
        "-L",
        "-o", str(output_file),
        url
    ]

    subprocess.run(cmd)
    return True


def check_torrent_option():
    print("\nChecking for torrent option...")
    print("Torrent downloads are often fastest but FMA doesn't officially provide torrents")
    print("Check: https://github.com/mdeff/fma/issues for community torrent links")


def main():
    print("="*70)
    print(" "*15 + "FMA MEDIUM FAST DOWNLOAD OPTIONS")
    print("="*70)

    print("\nOption 1: aria2c (RECOMMENDED - Multi-connection)")
    print("- Uses 16 parallel connections")
    print("- Usually 5-10x faster than single connection")
    print("- Resume support")

    print("\nOption 2: Kaggle")
    print("- Often has better bandwidth")
    print("- Requires kaggle account + API token")

    print("\nOption 3: curl with resume")
    print("- Can pause and resume")
    print("- Better than basic download")

    choice = input("\nChoose option (1/2/3): ").strip()

    if choice == "1":
        success = download_via_aria2c()
        if not success:
            print("\nFalling back to curl...")
            download_via_curl()
    elif choice == "2":
        download_via_kaggle()
    elif choice == "3":
        download_via_curl()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
