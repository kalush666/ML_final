import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from typing import Tuple, Dict, List, Optional
from collections import Counter
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.audio_features import AudioFeatureExtractor


def _extract_track_features(args):
    track_id, audio_path, artist_idx, n_segments, sample_rate, duration = args
    try:
        extractor = AudioFeatureExtractor(sample_rate=sample_rate, duration=duration)
        audio = extractor.load_audio(str(audio_path))
        features = extractor.extract_mel_spectrogram(audio)
        if features is not None:
            total_frames = features.shape[0]
            segment_length = total_frames // n_segments
            segments = []
            for i in range(n_segments):
                start = i * segment_length
                end = start + segment_length
                segment = features[start:end]
                segments.append((segment, artist_idx))
            return segments
    except Exception:
        pass
    return []


class FMAAuthorDataLoader:

    def __init__(self,
                 fma_audio_path: Path = Path('data/raw/fma_medium/fma_medium'),
                 fma_metadata_path: Path = Path('data/raw/fma_metadata/tracks.csv'),
                 min_tracks_per_artist: int = 10,
                 max_artists: int = 100,
                 sample_rate: int = 22050,
                 duration: float = 30.0,
                 num_workers: int = None):
        self.fma_audio_path = fma_audio_path
        self.fma_metadata_path = fma_metadata_path
        self.min_tracks_per_artist = min_tracks_per_artist
        self.max_artists = max_artists
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_workers = num_workers or max(1, cpu_count() - 1)

        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            duration=duration
        )

        self.tracks_df = None
        self.artist_to_idx = {}
        self.idx_to_artist = {}

    def load_metadata(self) -> pd.DataFrame:
        print("Loading FMA metadata...")
        self.tracks_df = pd.read_csv(
            self.fma_metadata_path,
            index_col=0,
            header=[0, 1]
        )
        return self.tracks_df

    def get_audio_path(self, track_id: int) -> Path:
        folder = f"{track_id:06d}"[:3]
        filename = f"{track_id:06d}.mp3"
        return self.fma_audio_path / folder / filename

    def select_artists(self) -> Dict[str, List[int]]:
        if self.tracks_df is None:
            self.load_metadata()

        artist_name = self.tracks_df[('artist', 'name')]
        subset = self.tracks_df[('set', 'subset')]
        medium_mask = subset == 'medium'

        medium_artists = artist_name[medium_mask]
        artist_counts = Counter(medium_artists)

        eligible_artists = [
            (artist, count) for artist, count in artist_counts.items()
            if count >= self.min_tracks_per_artist and pd.notna(artist)
        ]

        eligible_artists.sort(key=lambda x: x[1], reverse=True)
        selected = eligible_artists[:self.max_artists]

        print(f"Selected {len(selected)} artists with {self.min_tracks_per_artist}+ tracks")

        artist_tracks = {}
        for artist, _ in selected:
            track_ids = self.tracks_df[
                (artist_name == artist) & medium_mask
            ].index.tolist()

            available_tracks = []
            for tid in track_ids:
                if self.get_audio_path(tid).exists():
                    available_tracks.append(tid)

            if len(available_tracks) >= self.min_tracks_per_artist:
                artist_tracks[artist] = available_tracks

        self.artist_to_idx = {artist: i for i, artist in enumerate(artist_tracks.keys())}
        self.idx_to_artist = {i: artist for artist, i in self.artist_to_idx.items()}

        print(f"Artists with available audio: {len(artist_tracks)}")
        return artist_tracks

    def extract_features_for_tracks(self,
                                    track_ids: List[int],
                                    n_segments: int = 5) -> Optional[np.ndarray]:
        features_list = []

        for tid in track_ids:
            audio_path = self.get_audio_path(tid)
            try:
                features = self.feature_extractor.extract_mel_spectrogram(str(audio_path))
                if features is not None:
                    total_frames = features.shape[0]
                    segment_length = total_frames // n_segments

                    for i in range(n_segments):
                        start = i * segment_length
                        end = start + segment_length
                        segment = features[start:end]
                        features_list.append(segment)

            except Exception as e:
                print(f"Error extracting features for track {tid}: {e}")
                continue

        if not features_list:
            return None

        return np.array(features_list)

    def prepare_dataset(self,
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        n_segments: int = 5,
                        use_parallel: bool = True) -> Tuple:
        artist_tracks = self.select_artists()

        args_list = []
        for artist, track_ids in artist_tracks.items():
            artist_idx = self.artist_to_idx[artist]
            for tid in track_ids:
                audio_path = self.get_audio_path(tid)
                args_list.append((
                    tid, audio_path, artist_idx, n_segments,
                    self.sample_rate, self.duration
                ))

        all_features = []
        all_labels = []

        total_tracks = len(args_list)

        if use_parallel and total_tracks > 50:
            print(f"\nUsing {self.num_workers} parallel workers for feature extraction...")
            processed = 0

            with Pool(processes=self.num_workers) as pool:
                for segments in pool.imap_unordered(_extract_track_features, args_list):
                    for segment, label in segments:
                        all_features.append(segment)
                        all_labels.append(label)

                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed {processed}/{total_tracks} tracks...")

            print(f"  Completed {processed}/{total_tracks} tracks")
        else:
            for artist, track_ids in artist_tracks.items():
                print(f"Processing {artist} ({len(track_ids)} tracks)...")
                artist_idx = self.artist_to_idx[artist]

                for tid in track_ids:
                    audio_path = self.get_audio_path(tid)
                    try:
                        audio = self.feature_extractor.load_audio(str(audio_path))
                        features = self.feature_extractor.extract_mel_spectrogram(audio)
                        if features is not None:
                            total_frames = features.shape[0]
                            segment_length = total_frames // n_segments

                            for i in range(n_segments):
                                start = i * segment_length
                                end = start + segment_length
                                segment = features[start:end]
                                all_features.append(segment)
                                all_labels.append(artist_idx)

                    except Exception:
                        continue

        X = np.array(all_features)
        y = np.array(all_labels)

        print(f"\nTotal samples: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Number of classes: {len(self.artist_to_idx)}")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )

        from tensorflow import keras
        num_classes = len(self.artist_to_idx)
        y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes=num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_artist_name(self, idx: int) -> str:
        return self.idx_to_artist.get(idx, "Unknown")

    def get_num_classes(self) -> int:
        return len(self.artist_to_idx)
