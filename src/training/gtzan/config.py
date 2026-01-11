from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


GENRE_LABELS_10 = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

GENRE_LABELS_8 = [
    "blues", "classical", "country", "disco", 
    "hiphop", "jazz", "metal", "reggae"
]

EXCLUDED_GENRES = ["rock", "pop"]


@dataclass
class GTZANConfig:
    train_csv_path: Path = Path('data/processed/gtzan_splits/train.csv')
    validation_csv_path: Path = Path('data/processed/gtzan_splits/val.csv')
    test_csv_path: Path = Path('data/processed/gtzan_splits/test.csv')
    model_output_directory: Path = Path('models/gtzan_classifier')
    
    training_epochs: int = 100
    batch_size: int = 24
    initial_learning_rate: float = 0.0003
    segments_per_track: int = 5
    
    dropout_rate: float = 0.5
    l2_regularization: float = 0.0001
    focal_loss_gamma: float = 2.0
    label_smoothing_factor: float = 0.1
    
    enable_mixup_augmentation: bool = True
    mixup_alpha: float = 0.3
    enable_specaugment: bool = True
    freq_mask_param: int = 10
    time_mask_param: int = 30
    enable_test_time_augmentation: bool = True

    use_adaptive_focal_loss: bool = False
    confidence_penalty: float = 0.15
    per_class_gamma: Optional[List[float]] = None
    
    @property
    def number_of_classes(self) -> int:
        return len(GENRE_LABELS_10)
    
    @property
    def genre_names(self) -> list:
        return GENRE_LABELS_10
    
    def validate_paths(self) -> bool:
        return all([
            self.train_csv_path.exists(),
            self.validation_csv_path.exists(),
            self.test_csv_path.exists()
        ])
