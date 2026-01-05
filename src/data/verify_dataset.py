from pathlib import Path
import pandas as pd


class DatasetVerifier:

    def __init__(self):
        self.data_dir = Path("data/raw")
        self.metadata_dir = self.data_dir / "fma_metadata"
        self.audio_dir = self.data_dir / "fma_medium"
        self.errors = []
        self.warnings = []

    def verify_metadata(self) -> bool:
        print("Verifying metadata files...")

        if not self.metadata_dir.exists():
            self.errors.append(f"Metadata directory not found: {self.metadata_dir}")
            return False

        required_files = ["tracks.csv", "genres.csv"]
        for file in required_files:
            file_path = self.metadata_dir / file
            if not file_path.exists():
                self.errors.append(f"Missing metadata file: {file}")
                return False

        print("✓ Metadata files exist")
        return True

    def verify_audio_directory(self) -> bool:
        print("Verifying audio directory...")

        if not self.audio_dir.exists():
            self.errors.append(f"Audio directory not found: {self.audio_dir}")
            self.errors.append("Please download fma_medium.zip manually")
            return False

        subdirs = list(self.audio_dir.glob("*"))
        if len(subdirs) == 0:
            self.errors.append("Audio directory is empty")
            return False

        print(f"✓ Audio directory exists with {len(subdirs)} subdirectories")
        return True

    def count_audio_files(self) -> int:
        print("Counting audio files...")
        mp3_files = list(self.audio_dir.glob("**/*.mp3"))
        count = len(mp3_files)
        print(f"✓ Found {count} audio files")

        if count < 20000:
            self.warnings.append(f"Expected ~25000 files, found {count}")

        return count

    def verify_metadata_content(self) -> bool:
        print("Verifying metadata content...")

        try:
            tracks = pd.read_csv(
                self.metadata_dir / "tracks.csv",
                index_col=0,
                header=[0, 1]
            )

            medium_tracks = tracks[tracks[('set', 'subset')] == 'medium']
            print(f"✓ Found {len(medium_tracks)} tracks in medium subset")

            return True

        except Exception as e:
            self.errors.append(f"Failed to read metadata: {e}")
            return False

    def run_verification(self) -> bool:
        print("\n" + "="*60)
        print("Dataset Verification")
        print("="*60 + "\n")

        metadata_ok = self.verify_metadata()
        audio_ok = self.verify_audio_directory()

        if not metadata_ok or not audio_ok:
            self.print_results()
            return False

        self.count_audio_files()
        self.verify_metadata_content()

        self.print_results()

        return len(self.errors) == 0

    def print_results(self):
        print("\n" + "="*60)
        print("Verification Results")
        print("="*60)

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")
            print("\nReady to run: python src/data/pipeline.py")
        elif not self.errors:
            print("\n✅ Verification passed with warnings")
            print("\nReady to run: python src/data/pipeline.py")
        else:
            print("\n❌ Verification failed")
            print("\nPlease fix the errors above before proceeding")

        print("="*60 + "\n")


if __name__ == "__main__":
    verifier = DatasetVerifier()
    success = verifier.run_verification()
    exit(0 if success else 1)
