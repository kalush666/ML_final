import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict


class ClassBalanceAnalyzer:

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_genre_balance(self,
                             df: pd.DataFrame,
                             genre_column: Tuple[str, str]) -> Dict:
        genre_counts = df[genre_column].value_counts()

        balance_metrics = {
            'counts': genre_counts.to_dict(),
            'total': len(df),
            'num_classes': len(genre_counts),
            'min_samples': genre_counts.min(),
            'max_samples': genre_counts.max(),
            'mean_samples': genre_counts.mean(),
            'imbalance_ratio': genre_counts.max() / genre_counts.min()
        }

        return balance_metrics

    def analyze_artist_balance(self,
                               df: pd.DataFrame,
                               artist_column: Tuple[str, str],
                               top_n: int = 50) -> Dict:
        artist_counts = df[artist_column].value_counts()

        balance_metrics = {
            'total_artists': len(artist_counts),
            'top_artist_samples': artist_counts.iloc[0] if len(artist_counts) > 0 else 0,
            'mean_samples_per_artist': artist_counts.mean(),
            'median_samples_per_artist': artist_counts.median(),
            'artists_with_single_track': (artist_counts == 1).sum()
        }

        return balance_metrics

    def plot_genre_distribution(self,
                               df: pd.DataFrame,
                               genre_column: Tuple[str, str],
                               save: bool = True) -> None:
        genre_counts = df[genre_column].value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
        plt.xlabel('Number of Tracks')
        plt.ylabel('Genre')
        plt.title('Genre Distribution')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'genre_distribution.png')
        plt.close()

    def plot_artist_distribution(self,
                                df: pd.DataFrame,
                                artist_column: Tuple[str, str],
                                top_n: int = 30,
                                save: bool = True) -> None:
        artist_counts = df[artist_column].value_counts().head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=artist_counts.values, y=artist_counts.index, palette='rocket')
        plt.xlabel('Number of Tracks')
        plt.ylabel('Artist')
        plt.title(f'Top {top_n} Artists by Track Count')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'artist_distribution.png')
        plt.close()

    def check_augmentation_needed(self,
                                 balance_metrics: Dict,
                                 threshold: float = 3.0) -> bool:
        imbalance_ratio = balance_metrics.get('imbalance_ratio', 1.0)
        return imbalance_ratio > threshold
