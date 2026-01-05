import pandas as pd
from pathlib import Path
from typing import Tuple, Dict


class FMADataExplorer:

    def __init__(self, metadata_path: Path):
        self.metadata_path = Path(metadata_path)
        self.tracks = None
        self.genres = None

    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tracks_file = self.metadata_path / "tracks.csv"
        genres_file = self.metadata_path / "genres.csv"

        self.tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
        self.genres = pd.read_csv(genres_file, index_col=0)

        return self.tracks, self.genres

    def get_genre_distribution(self) -> pd.Series:
        genre_top = self.tracks[('track', 'genre_top')]
        return genre_top.value_counts()

    def get_artist_distribution(self) -> pd.Series:
        artist = self.tracks[('artist', 'name')]
        return artist.value_counts()

    def get_dataset_summary(self) -> Dict:
        return {
            'total_tracks': len(self.tracks),
            'unique_genres': self.tracks[('track', 'genre_top')].nunique(),
            'unique_artists': self.tracks[('artist', 'name')].nunique(),
            'missing_genre': self.tracks[('track', 'genre_top')].isna().sum(),
            'missing_artist': self.tracks[('artist', 'name')].isna().sum()
        }

    def filter_medium_subset(self) -> pd.DataFrame:
        subset = self.tracks[('set', 'subset')]
        return self.tracks[subset == 'medium']
