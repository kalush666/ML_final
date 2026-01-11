
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.gtzan import GTZANConfig, GTZANTrainingOrchestrator


def main():
    import multiprocessing
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = GTZANConfig(
        train_csv_path=Path('data/processed/gtzan_splits/train_fixed.csv'),
        validation_csv_path=Path('data/processed/gtzan_splits/val_fixed.csv'),
        test_csv_path=Path('data/processed/gtzan_splits/test_fixed.csv'),
        model_output_directory=Path('models/gtzan_classifier_v2'),

        training_epochs=100,
        batch_size=16,
        initial_learning_rate=0.0003,
        segments_per_track=5,

        dropout_rate=0.5,
        l2_regularization=0.0001,

        focal_loss_gamma=2.0,
        label_smoothing_factor=0.1,

        enable_mixup_augmentation=True
    )

    if not config.validate_paths():
        print("\nGTZAN splits not found!")
        print("Run first: python src/data/download_gtzan.py")
        sys.exit(1)

    print(f"\nSystem Configuration:")
    print(f"  CPU cores: {multiprocessing.cpu_count()}")
    print(f"  Mode: CPU-only training")

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Focal loss gamma: {config.focal_loss_gamma}")
    print(f"  Label smoothing: {config.label_smoothing_factor}")
    print(f"  Dropout: {config.dropout_rate}")

    orchestrator = GTZANTrainingOrchestrator(config)
    orchestrator.execute_full_pipeline()


if __name__ == '__main__':
    main()
