import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tensorflow import keras

from utils.audio_features import AudioFeatureExtractor
from .config import GTZANConfig


class GTZANDatasetLoader:

    def __init__(self, config: GTZANConfig, include_rhythm_features: bool = True,
                 include_rock_features: bool = False):
        self.config = config
        self.include_rhythm_features = include_rhythm_features
        self.include_rock_features = include_rock_features
        self.audio_feature_extractor = AudioFeatureExtractor(
            include_rhythm=include_rhythm_features,
            include_rock_features=include_rock_features
        )
        self.genre_to_index_mapping = {
            genre: idx for idx, genre in enumerate(config.genre_names)
        }

        self.training_features = None
        self.training_labels = None
        self.validation_features = None
        self.validation_labels = None
        self.test_features = None
        self.test_labels = None
        self.computed_class_weights = None

        if include_rhythm_features and include_rock_features:
            print("  Feature extraction: Rhythm + rock features (167 dims)")
        elif include_rhythm_features:
            print("  Feature extraction: Rhythm features only (164 dims)")
        else:
            print("  Feature extraction: Standard features only (160 dims)")
        
    def load_all_datasets(self):
        self._load_training_data()
        self._load_validation_data()
        self._load_test_data()
        self._compute_class_weights()
        
    def _load_training_data(self):
        print("\nLoading training data...")
        features, labels = self._extract_features_from_csv(
            self.config.train_csv_path,
            segments_per_track=self.config.segments_per_track
        )
        self.training_features = features
        self.training_labels = labels
        print(f"  Training samples: {len(self.training_features)}")
        
    def _load_validation_data(self):
        print("\nLoading validation data...")
        features, labels = self._extract_features_from_csv(
            self.config.validation_csv_path,
            segments_per_track=1
        )
        self.validation_features = features
        self.validation_labels = labels
        print(f"  Validation samples: {len(self.validation_features)}")
        
    def _load_test_data(self):
        print("\nLoading test data...")
        features, labels = self._extract_features_from_csv(
            self.config.test_csv_path,
            segments_per_track=1
        )
        self.test_features = features
        self.test_labels = labels
        print(f"  Test samples: {len(self.test_features)}")
        
    def _extract_features_from_csv(self, csv_path: Path, segments_per_track: int):
        dataframe = pd.read_csv(csv_path)
        extracted_features = []
        extracted_labels = []
        
        total_tracks = len(dataframe)
        
        for row_index, row in dataframe.iterrows():
            audio_file_path = Path(row['path'])
            genre_name = row['genre']
            
            if genre_name not in self.genre_to_index_mapping:
                continue
                
            genre_index = self.genre_to_index_mapping[genre_name]
            
            try:
                track_features = self._extract_track_features(
                    audio_file_path, 
                    segments_per_track
                )
                
                for feature_array in track_features:
                    extracted_features.append(feature_array)
                    extracted_labels.append(genre_index)
                    
            except Exception as extraction_error:
                print(f"Error loading {audio_file_path}: {extraction_error}")
                continue
            
            if (row_index + 1) % 100 == 0:
                print(f"  Processed {row_index + 1}/{total_tracks} tracks...")
        
        feature_array = np.array(extracted_features)
        label_array = keras.utils.to_categorical(
            extracted_labels, 
            num_classes=self.config.number_of_classes
        )
        
        return feature_array, label_array
    
    def _extract_track_features(self, audio_path: Path, segments_count: int):
        if segments_count > 1:
            segment_features = self.audio_feature_extractor.extract_multi_segments(
                audio_path, 
                n_segments=segments_count, 
                random_offset=True
            )
            if segment_features is not None:
                return segment_features
            return []
        else:
            audio_signal = self.audio_feature_extractor.load_audio(audio_path)
            combined_features = self.audio_feature_extractor.extract_combined_features(audio_signal)
            return [combined_features]
    
    def _compute_class_weights(self):
        label_indices = np.argmax(self.training_labels, axis=1)
        class_sample_counts = Counter(label_indices)
        total_samples = len(label_indices)
        
        base_weights = {
            class_index: np.sqrt(total_samples / (self.config.number_of_classes * sample_count))
            for class_index, sample_count in class_sample_counts.items()
        }
        
        strategic_multipliers = {
            0: 0.7, 1: 1.0, 2: 1.0, 3: 1.3, 4: 1.0,
            5: 1.0, 6: 1.0, 7: 1.3, 8: 1.0, 9: 1.5
        }
        
        self.computed_class_weights = {
            class_index: base_weights[class_index] * strategic_multipliers.get(class_index, 1.0)
            for class_index in base_weights
        }
        
        self._print_class_distribution(class_sample_counts)
        
    def _print_class_distribution(self, class_counts: Counter):
        print("\nClass distribution (training):")
        for class_index, genre_name in enumerate(self.config.genre_names):
            sample_count = class_counts.get(class_index, 0)
            weight_value = self.computed_class_weights.get(class_index, 1.0)
            print(f"  {genre_name}: {sample_count} samples, weight={weight_value:.2f}")
            
    def get_input_shape(self):
        return (self.training_features.shape[1], self.training_features.shape[2])
    
    def get_training_data(self):
        return self.training_features, self.training_labels
    
    def get_validation_data(self):
        return self.validation_features, self.validation_labels
    
    def get_test_data(self):
        return self.test_features, self.test_labels
