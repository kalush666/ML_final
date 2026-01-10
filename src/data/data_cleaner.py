import librosa
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioDataCleaner:

    def __init__(self, audio_dir: Path):
        self.audio_dir = Path(audio_dir)

    def validate_audio_file(self, file_path: Path) -> bool:
        try:
            audio, sr = librosa.load(file_path, duration=3.0)
            return len(audio) > 0
        except Exception as e:
            logger.warning(f"Corrupted file {file_path}: {e}")
            return False

    def get_corrupted_files(self, track_ids: List[int]) -> List[int]:
        corrupted = []

        for track_id in track_ids:
            audio_path = self._get_audio_path(track_id)

            if not audio_path.exists():
                corrupted.append(track_id)
                continue

            if not self.validate_audio_file(audio_path):
                corrupted.append(track_id)

        return corrupted

    def _get_audio_path(self, track_id: int) -> Path:
        tid_str = f"{track_id:06d}"
        return self.audio_dir / tid_str[:3] / f"{tid_str}.mp3"


class MetadataCleaner:

    @staticmethod
    def remove_missing_labels(df: pd.DataFrame,
                             genre_column: Tuple[str, str],
                             artist_column: Tuple[str, str]) -> pd.DataFrame:
        clean_df = df.dropna(subset=[genre_column, artist_column])
        return clean_df

    @staticmethod
    def remove_corrupted_tracks(df: pd.DataFrame,
                               corrupted_ids: List[int]) -> pd.DataFrame:
        return df.drop(corrupted_ids, errors='ignore')

    @staticmethod
    def filter_min_tracks_per_artist(df: pd.DataFrame,
                                    artist_column: Tuple[str, str],
                                    min_tracks: int = 5,
                                    protected_genres: List[str] = None,
                                    genre_column: Tuple[str, str] = None) -> pd.DataFrame:
        """Filter artists with too few tracks, but protect minority genres."""
        if protected_genres and genre_column:
            protected_mask = df[genre_column].isin(protected_genres)
            protected_df = df[protected_mask]
            unprotected_df = df[~protected_mask]
            
            artist_counts = unprotected_df[artist_column].value_counts()
            valid_artists = artist_counts[artist_counts >= min_tracks].index
            filtered_unprotected = unprotected_df[unprotected_df[artist_column].isin(valid_artists)]
            
            return pd.concat([protected_df, filtered_unprotected])
        else:
            artist_counts = df[artist_column].value_counts()
            valid_artists = artist_counts[artist_counts >= min_tracks].index
            return df[df[artist_column].isin(valid_artists)]
