
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
        model_output_directory=Path('models/gtzan_classifier_v6'),

        training_epochs=150,
        batch_size=20,
        initial_learning_rate=0.00013,
        segments_per_track=5,

        dropout_rate=0.56,
        l2_regularization=0.00022,

        focal_loss_gamma=2.5,
        label_smoothing_factor=0.09,

        enable_mixup_augmentation=True,
        mixup_alpha=0.35,
        use_adaptive_focal_loss=True,
        confidence_penalty=0.20,
        per_class_gamma=[2.0, 2.0, 3.0, 4.0, 2.5, 2.0, 2.5, 5.5, 2.5, 8.0],

        enable_specaugment=True,
        freq_mask_param=12,
        time_mask_param=35
    )

    if not config.validate_paths():
        print("\nGTZAN splits not found!")
        print("Run first: python src/data/download_gtzan.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("GTZAN V6 Training - Balanced Regularization")
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

    v6_checkpoint_path = config.model_output_directory / 'checkpoint_best.keras'
    pretrained_v4_path = Path('models/gtzan_classifier_v4/gtzan_classifier_final.keras')

    if v6_checkpoint_path.exists():
        print(f"\nResuming Training:")
        print(f"  Loading V6 checkpoint from previous run")
        pretrained_model = v6_checkpoint_path
    elif pretrained_v4_path.exists():
        print(f"\nTransfer Learning:")
        print(f"  Loading weights from V4 (best model: 76% accuracy)")
        pretrained_model = pretrained_v4_path
    else:
        pretrained_model = None
        print(f"\nStarting from random initialization")

    orchestrator = GTZANTrainingOrchestrator(config, include_rhythm_features=True, pretrained_model_path=pretrained_model)
    orchestrator.execute_full_pipeline()


if __name__ == '__main__':
    main()
