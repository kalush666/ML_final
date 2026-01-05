import json
from pathlib import Path
from data_explorer import FMADataExplorer
from data_cleaner import AudioDataCleaner, MetadataCleaner
from data_analyzer import ClassBalanceAnalyzer
from data_augmenter import AudioAugmenter, DatasetAugmenter
from data_splitter import DatasetSplitter


class DataPreparationPipeline:

    def __init__(self,
                 metadata_path: Path,
                 audio_dir: Path,
                 output_dir: Path):
        self.metadata_path = Path(metadata_path)
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.explorer = FMADataExplorer(metadata_path)
        self.audio_cleaner = AudioDataCleaner(audio_dir)
        self.metadata_cleaner = MetadataCleaner()
        self.analyzer = ClassBalanceAnalyzer(output_dir / 'analysis')
        self.augmenter = DatasetAugmenter(AudioAugmenter())
        self.splitter = DatasetSplitter()

    def run(self) -> None:
        print("Step 1: Loading and exploring metadata...")
        tracks, genres = self.explorer.load_metadata()
        summary = self.explorer.get_dataset_summary()
        self._save_json(summary, 'dataset_summary.json')
        print(f"Dataset summary: {summary}")

        print("\nStep 2: Filtering medium subset...")
        medium_tracks = self.explorer.filter_medium_subset()
        print(f"Medium subset size: {len(medium_tracks)}")

        print("\nStep 3: Cleaning data...")
        genre_col = ('track', 'genre_top')
        artist_col = ('artist', 'name')

        clean_tracks = self.metadata_cleaner.remove_missing_labels(
            medium_tracks, genre_col, artist_col
        )
        print(f"After removing missing labels: {len(clean_tracks)}")

        print("\nStep 4: Validating audio files...")
        track_ids = clean_tracks.index.tolist()[:100]
        corrupted = self.audio_cleaner.get_corrupted_files(track_ids)
        print(f"Found {len(corrupted)} corrupted files in sample")

        clean_tracks = self.metadata_cleaner.remove_corrupted_tracks(
            clean_tracks, corrupted
        )

        clean_tracks = self.metadata_cleaner.filter_min_tracks_per_artist(
            clean_tracks, artist_col, min_tracks=5
        )
        print(f"After filtering artists: {len(clean_tracks)}")

        print("\nStep 5: Analyzing class balance...")
        genre_balance = self.analyzer.analyze_genre_balance(clean_tracks, genre_col)
        artist_balance = self.analyzer.analyze_artist_balance(clean_tracks, artist_col)

        self._save_json(genre_balance, 'genre_balance.json')
        self._save_json(artist_balance, 'artist_balance.json')

        print(f"Genre balance: {genre_balance}")
        print(f"Artist balance: {artist_balance}")

        self.analyzer.plot_genre_distribution(clean_tracks, genre_col)
        self.analyzer.plot_artist_distribution(clean_tracks, artist_col)

        print("\nStep 6: Checking if augmentation is needed...")
        needs_augmentation = self.analyzer.check_augmentation_needed(genre_balance)
        print(f"Augmentation needed: {needs_augmentation}")

        print("\nStep 7: Creating train/val/test splits...")
        train_df, val_df, test_df = self.splitter.split_data(clean_tracks)
        split_summary = self.splitter.get_split_summary(train_df, val_df, test_df)

        self._save_json(split_summary, 'split_summary.json')
        print(f"Split summary: {split_summary}")

        self.splitter.save_splits(train_df, val_df, test_df, self.output_dir / 'splits')

        print("\nData preparation pipeline completed!")

    def _save_json(self, data: dict, filename: str) -> None:
        with open(self.output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    pipeline = DataPreparationPipeline(
        metadata_path=Path("data/raw/fma_metadata"),
        audio_dir=Path("data/raw/fma_medium"),
        output_dir=Path("data/processed")
    )

    pipeline.run()
