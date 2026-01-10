"""
GTZAN Dataset Downloader
========================
Downloads and prepares the GTZAN music genre dataset.
- 10 genres, 100 tracks each (1000 total)
- 30 seconds per track
- ~1.2GB download
"""

import os
import random
import csv
from pathlib import Path


GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


def verify_gtzan(gtzan_dir: Path) -> bool:
    """Verify GTZAN dataset structure."""
    print("\n" + "="*60)
    print("Verifying GTZAN Dataset")
    print("="*60)
    
    # Check for genres folder (might be nested)
    genres_dir = gtzan_dir / "genres" if (gtzan_dir / "genres").exists() else gtzan_dir
    
    # Also check for Kaggle structure
    if (gtzan_dir / "Data" / "genres_original").exists():
        genres_dir = gtzan_dir / "Data" / "genres_original"
    
    missing = []
    counts = {}
    
    for genre in GTZAN_GENRES:
        genre_path = genres_dir / genre
        if not genre_path.exists():
            missing.append(genre)
        else:
            wav_files = list(genre_path.glob("*.wav")) + list(genre_path.glob("*.au"))
            counts[genre] = len(wav_files)
    
    if missing:
        print(f"✗ Missing genres: {missing}")
        print(f"  Looked in: {genres_dir}")
        return False
    
    print(f"Found genres in: {genres_dir}")
    print("\nGenre distribution:")
    total = 0
    for genre, count in counts.items():
        print(f"  {genre}: {count} tracks")
        total += count
    
    print(f"\nTotal: {total} tracks")
    
    if total >= 900:  # Allow some tolerance
        print("✓ Dataset verification passed")
        return True
    else:
        print("✗ Dataset incomplete")
        return False


def find_genres_dir(gtzan_dir: Path) -> Path:
    """Find the actual genres directory."""
    if (gtzan_dir / "genres").exists():
        return gtzan_dir / "genres"
    if (gtzan_dir / "Data" / "genres_original").exists():
        return gtzan_dir / "Data" / "genres_original"
    return gtzan_dir


def create_gtzan_splits(gtzan_dir: Path, output_dir: Path, 
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15) -> None:
    """Create train/val/test splits for GTZAN."""
    print("\n" + "="*60)
    print("Creating Train/Val/Test Splits")
    print("="*60)
    
    genres_dir = find_genres_dir(gtzan_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_tracks = []
    
    for genre in GTZAN_GENRES:
        genre_path = genres_dir / genre
        if not genre_path.exists():
            continue
            
        for audio_file in genre_path.glob("*.wav"):
            all_tracks.append({
                "path": str(audio_file.absolute()),
                "genre": genre,
                "filename": audio_file.name
            })
        # Also check .au files
        for audio_file in genre_path.glob("*.au"):
            all_tracks.append({
                "path": str(audio_file.absolute()),
                "genre": genre,
                "filename": audio_file.name
            })
    
    print(f"Found {len(all_tracks)} total tracks")
    
    # Split per genre to maintain balance
    train_data, val_data, test_data = [], [], []
    
    for genre in GTZAN_GENRES:
        genre_tracks = [t for t in all_tracks if t["genre"] == genre]
        random.seed(42)
        random.shuffle(genre_tracks)
        
        n = len(genre_tracks)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data.extend(genre_tracks[:n_train])
        val_data.extend(genre_tracks[n_train:n_train + n_val])
        test_data.extend(genre_tracks[n_train + n_val:])
    
    # Save splits
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        csv_path = output_dir / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "genre", "filename"])
            writer.writeheader()
            writer.writerows(data)
        print(f"  {name}: {len(data)} tracks -> {csv_path}")
    
    print("\n✓ Splits created successfully")


def main():
    print("\n" + "="*70)
    print(" " * 20 + "GTZAN DATASET SETUP")
    print("="*70)
    
    gtzan_dir = Path("data/raw/gtzan")
    splits_dir = Path("data/processed/gtzan_splits")
    
    # Check if already downloaded
    if verify_gtzan(gtzan_dir):
        print("\nGTZAN downloaded and verified!")
        create_gtzan_splits(gtzan_dir, splits_dir)
        
        print("\n" + "="*70)
        print(" " * 25 + "✅ SETUP COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  Run: .\\.\\.venv311\\Scripts\\python.exe src/training/train_gtzan.py")
        print("="*70 + "\n")
    else:
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*60)
        print("\nOption 1: Kaggle (recommended)")
        print("  1. Visit: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
        print("  2. Download and extract to: data/raw/gtzan/")
        print("\nOption 2: HuggingFace")
        print("  Run in PowerShell:")
        print('  Invoke-WebRequest -Uri "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz" -OutFile "data/raw/gtzan/genres.tar.gz"')
        print("  cd data/raw/gtzan; tar -xzf genres.tar.gz")


if __name__ == "__main__":
    main()
