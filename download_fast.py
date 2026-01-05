import subprocess
from pathlib import Path
import sys


def install_aria2():
    print("Installing aria2 for fast multi-connection download...")
    try:
        subprocess.run(["winget", "install", "-e", "--id", "aria2.aria2"], check=True)
        print("✓ aria2 installed")
        return True
    except:
        print("✗ Failed to install aria2 automatically")
        print("Install manually: winget install aria2.aria2")
        return False


def download_with_aria2():
    url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Starting FMA Medium download with aria2 (16 connections)")
    print("="*70)
    print(f"\nURL: {url}")
    print(f"Output: {output_dir}/fma_medium.zip")
    print(f"Size: ~25GB")
    print("\nThis should be 5-10x faster than single connection!")
    print("Press Ctrl+C to pause (can resume later)\n")

    cmd = [
        "aria2c",
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        "--file-allocation=none",
        "--continue=true",
        "--dir", str(output_dir),
        "--out", "fma_medium.zip",
        url
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Download completed!")
        print(f"\nNext steps:")
        print("1. Extract fma_medium.zip to data/raw/fma_medium/")
        print("2. Run: python src/data/verify_dataset.py")
        return True
    except FileNotFoundError:
        print("\n✗ aria2c not found!")
        print("\nInstall it first:")
        print("  winget install aria2.aria2")
        print("\nOr download from: https://github.com/aria2/aria2/releases")
        return False
    except subprocess.CalledProcessError:
        print("\n✗ Download failed or cancelled")
        print("Run again to resume from where it stopped")
        return False


if __name__ == "__main__":
    print("Checking for aria2...")
    check = subprocess.run(["aria2c", "--version"], capture_output=True, shell=True)

    if check.returncode != 0:
        print("aria2c not found in PATH")
        print("Please restart your terminal/PowerShell for PATH changes to take effect")
        print("\nThen run this script again")
        sys.exit(1)

    print("✓ aria2c found")
    download_with_aria2()
