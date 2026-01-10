import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.gtzan import GTZANConfig, GTZANTrainingOrchestrator


def main():
    config = GTZANConfig(
        train_csv_path=Path('data/processed/gtzan_splits/train.csv'),
        validation_csv_path=Path('data/processed/gtzan_splits/val.csv'),
        test_csv_path=Path('data/processed/gtzan_splits/test.csv'),
        model_output_directory=Path('models/gtzan_classifier_v2'),
        training_epochs=100,
        batch_size=24,
        initial_learning_rate=0.00015,
        segments_per_track=5,
        dropout_rate=0.65,
        l2_regularization=0.001,
        focal_loss_gamma=3.0,
        label_smoothing_factor=0.15,
        enable_mixup_augmentation=True
    )
    
    if not config.validate_paths():
        print("GTZAN splits not found!")
        print("Run first: python src/data/download_gtzan.py")
        sys.exit(1)
    
    orchestrator = GTZANTrainingOrchestrator(config)
    orchestrator.execute_full_pipeline()


if __name__ == '__main__':
    main()
