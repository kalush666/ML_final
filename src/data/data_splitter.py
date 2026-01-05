import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple


class DatasetSplitter:

    def __init__(self,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, temp_df = train_test_split(
            df,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_state,
            stratify=df[('track', 'genre_top')]
        )

        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)

        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.random_state,
            stratify=temp_df[('track', 'genre_top')]
        )

        return train_df, val_df, test_df

    def save_splits(self,
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_dir / 'train.csv')
        val_df.to_csv(output_dir / 'val.csv')
        test_df.to_csv(output_dir / 'test.csv')

    def get_split_summary(self,
                         train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> dict:
        return {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'total_size': len(train_df) + len(val_df) + len(test_df),
            'train_ratio': len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
            'val_ratio': len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
            'test_ratio': len(test_df) / (len(train_df) + len(val_df) + len(test_df))
        }
