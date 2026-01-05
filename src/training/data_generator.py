import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('src')
from utils.audio_features import AudioFeatureExtractor


class GenreDataGenerator:

    def __init__(self,
                 csv_path: Path,
                 audio_dir: Path,
                 feature_extractor: AudioFeatureExtractor,
                 batch_size: int = 32):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size

        self.df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
        self.label_encoder = LabelEncoder()

        self.prepare_data()

    def prepare_data(self) -> None:
        genre_col = ('track', 'genre_top')

        self.genres = self.df[genre_col].values
        self.track_ids = self.df.index.tolist()

        self.encoded_labels = self.label_encoder.fit_transform(self.genres)
        self.num_classes = len(self.label_encoder.classes_)

    def get_audio_path(self, track_id: int) -> Path:
        tid_str = f"{track_id:06d}"
        return self.audio_dir / tid_str[:3] / f"{tid_str}.mp3"

    def load_and_extract_features(self, track_id: int) -> np.ndarray:
        audio_path = self.get_audio_path(track_id)

        try:
            audio = self.feature_extractor.load_audio(audio_path)
            features = self.feature_extractor.extract_combined_features(audio)
            return features
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        features_list = []
        labels_list = []
        valid_indices = []

        for idx, track_id in enumerate(self.track_ids):
            features = self.load_and_extract_features(track_id)

            if features is not None:
                features_list.append(features)
                labels_list.append(self.encoded_labels[idx])
                valid_indices.append(idx)

            if len(features_list) % 100 == 0:
                print(f"Processed {len(features_list)} tracks...")

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        labels_categorical = keras.utils.to_categorical(labels_array, num_classes=self.num_classes)

        return features_array, labels_categorical, list(self.label_encoder.classes_)

    def get_class_names(self) -> List[str]:
        return list(self.label_encoder.classes_)

    def get_class_distribution(self) -> dict:
        unique, counts = np.unique(self.encoded_labels, return_counts=True)
        distribution = {
            self.label_encoder.classes_[idx]: int(count)
            for idx, count in zip(unique, counts)
        }
        return distribution

    def calculate_class_weights(self) -> dict:
        unique, counts = np.unique(self.encoded_labels, return_counts=True)
        total = len(self.encoded_labels)

        class_weights = {}
        for idx, count in zip(unique, counts):
            weight = total / (len(unique) * count)
            class_weights[int(idx)] = float(weight)

        return class_weights
