import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
sys.path.append('src')
from utils.audio_features import AudioFeatureExtractor


class GenreDataGenerator:

    def __init__(self,
                 csv_path: Path,
                 audio_dir: Path,
                 feature_extractor: AudioFeatureExtractor,
                 batch_size: int = 32,
                 num_workers: int = None,
                 n_segments: int = 1):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.n_segments = n_segments

        self.df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
        self.label_encoder = LabelEncoder()

        self.prepare_data()

    def prepare_data(self) -> None:
        genre_col = ('track', 'genre_top')

        self.genres = self.df[genre_col].values
        self.track_ids = self.df.index.tolist()

        if hasattr(self.label_encoder, 'classes_'):
            self.encoded_labels = self.label_encoder.transform(self.genres)
        else:
            self.encoded_labels = self.label_encoder.fit_transform(self.genres)
            self.num_classes = len(self.label_encoder.classes_)

    def get_audio_path(self, track_id: int) -> Path:
        tid_str = f"{track_id:06d}"
        return self.audio_dir / tid_str[:3] / f"{tid_str}.mp3"

    def load_and_extract_features(self, track_id: int) -> np.ndarray:
        audio_path = self.get_audio_path(track_id)

        try:
            if self.n_segments > 1:
                segments = self.feature_extractor.extract_multi_segments(
                    audio_path, n_segments=self.n_segments, random_offset=True)
                return segments
            else:
                audio = self.feature_extractor.load_audio(audio_path)
                features = self.feature_extractor.extract_combined_features(audio)
                return features
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

    @staticmethod
    def _process_track(args):
        track_id, audio_dir, feature_extractor = args
        tid_str = f"{track_id:06d}"
        audio_path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"

        try:
            audio = feature_extractor.load_audio(audio_path)
            features = feature_extractor.extract_combined_features(audio)
            return track_id, features
        except Exception:
            return track_id, None

    def generate_dataset(self, use_parallel: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if use_parallel and len(self.track_ids) > 100:
            return self._generate_dataset_parallel()
        else:
            return self._generate_dataset_sequential()

    def _generate_dataset_sequential(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        features_list = []
        labels_list = []
        valid_indices = []

        for idx, track_id in enumerate(self.track_ids):
            features = self.load_and_extract_features(track_id)

            if features is not None:
                if self.n_segments > 1:
                    for seg_features in features:
                        features_list.append(seg_features)
                        labels_list.append(self.encoded_labels[idx])
                else:
                    features_list.append(features)
                    labels_list.append(self.encoded_labels[idx])
                valid_indices.append(idx)

            if len(valid_indices) % 100 == 0:
                print(f"Processed {len(valid_indices)} tracks...")

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        labels_categorical = keras.utils.to_categorical(labels_array, num_classes=self.num_classes)

        return features_array, labels_categorical, list(self.label_encoder.classes_)

    def _generate_dataset_parallel(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print(f"Using {self.num_workers} parallel workers for feature extraction...")

        args_list = [(track_id, self.audio_dir, self.feature_extractor)
                     for track_id in self.track_ids]

        features_dict = {}
        processed_count = 0

        with Pool(processes=self.num_workers) as pool:
            for track_id, features in pool.imap_unordered(self._process_track, args_list):
                if features is not None:
                    features_dict[track_id] = features

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count}/{len(self.track_ids)} tracks...")

        features_list = []
        labels_list = []

        for idx, track_id in enumerate(self.track_ids):
            if track_id in features_dict:
                features_list.append(features_dict[track_id])
                labels_list.append(self.encoded_labels[idx])

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        labels_categorical = keras.utils.to_categorical(labels_array, num_classes=self.num_classes)

        print(f"Successfully extracted features from {len(features_list)}/{len(self.track_ids)} tracks")

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
