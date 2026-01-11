
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
        model_output_directory=Path('models/gtzan_classifier_v5'),

        training_epochs=120,
        batch_size=16,
        initial_learning_rate=0.0001,
        segments_per_track=5,

        dropout_rate=0.60,
        l2_regularization=0.0003,

        focal_loss_gamma=2.0,
        label_smoothing_factor=0.12,

        enable_mixup_augmentation=True,
        mixup_alpha=0.4,
        use_adaptive_focal_loss=True,
        confidence_penalty=0.25,
        per_class_gamma=[2.0, 2.0, 2.5, 3.5, 2.5, 2.0, 2.5, 6.0, 2.5, 6.0],

        enable_specaugment=True,
        freq_mask_param=15,
        time_mask_param=40
    )

    if not config.validate_paths():
        print("\nGTZAN splits not found!")
        print("Run first: python src/data/download_gtzan.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("GTZAN V5 Training - Enhanced Pop/Rock Discrimination")
    print(f"{'='*60}")
    
    print(f"\nSystem Configuration:")
    print(f"  CPU cores: {multiprocessing.cpu_count()}")

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.initial_learning_rate}")
    print(f"  Epochs: {config.training_epochs}")
    print(f"  Dropout: {config.dropout_rate}")
    
    print(f"\nLoss Configuration:")
    print(f"  Adaptive focal loss: {config.use_adaptive_focal_loss}")
    print(f"  Label smoothing: {config.label_smoothing_factor}")
    print(f"  Confidence penalty: {config.confidence_penalty}")

    orchestrator = GTZANTrainingOrchestrator(config, include_rhythm_features=True)
    orchestrator.execute_full_pipeline()


if __name__ == '__main__':
    main()
