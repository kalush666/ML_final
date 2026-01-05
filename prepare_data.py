import sys
from pathlib import Path

sys.path.append('src')

from data.download_metadata import download_metadata, print_download_instructions
from data.verify_dataset import DatasetVerifier
from data.pipeline import DataPreparationPipeline


def main():
    print("\n" + "="*70)
    print(" "*20 + "DATA PREPARATION WORKFLOW")
    print("="*70 + "\n")

    print("STAGE 1: Download Metadata")
    print("-" * 70)
    try:
        metadata_path = download_metadata()
        print(f"✓ Metadata ready at: {metadata_path}\n")
    except Exception as e:
        print(f"✗ Failed to download metadata: {e}")
        return

    print_download_instructions()

    print("\nSTAGE 2: Verify Dataset")
    print("-" * 70)
    verifier = DatasetVerifier()
    if not verifier.run_verification():
        print("\n⚠️  Please complete the manual download step and run again")
        print("Command: python prepare_data.py")
        return

    print("\nSTAGE 3: Run Data Preparation Pipeline")
    print("-" * 70)
    print("Starting comprehensive data preparation...\n")

    try:
        pipeline = DataPreparationPipeline(
            metadata_path=Path("data/raw/fma_metadata"),
            audio_dir=Path("data/raw/fma_medium"),
            output_dir=Path("data/processed")
        )
        pipeline.run()

        print("\n" + "="*70)
        print(" "*25 + "✅ SUCCESS!")
        print("="*70)
        print("\nData preparation completed successfully!")
        print("\nGenerated outputs:")
        print("  - data/processed/splits/        (train/val/test CSV files)")
        print("  - data/processed/analysis/      (distribution plots)")
        print("  - data/processed/*.json         (analysis reports)")
        print("\nNext step: Model training")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
