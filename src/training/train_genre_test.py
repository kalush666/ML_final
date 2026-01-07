import sys
from pathlib import Path
sys.path.append('src')

import numpy as np
import pandas as pd
from utils.audio_features import AudioFeatureExtractor
from training.data_generator import GenreDataGenerator
from models.genre_classifier import GenreCNNClassifier
from training.metrics import MetricsTracker


class QuickTestPipeline:

    def __init__(self,
                 train_csv: Path,
                 val_csv: Path,
                 test_csv: Path,
                 audio_dir: Path,
                 output_dir: Path,
                 sample_size: int = 100):
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size

        self.feature_extractor = AudioFeatureExtractor()

    def prepare_small_datasets(self):
        print(f"Creating small test dataset ({self.sample_size} tracks per split)...")

        train_gen = GenreDataGenerator(
            self.train_csv,
            self.audio_dir,
            self.feature_extractor
        )

        train_gen.df = train_gen.df.head(self.sample_size)
        train_gen.prepare_data()

        self.X_train, self.y_train, self.class_names = train_gen.generate_dataset()
        self.num_classes = len(self.class_names)
        self.class_weights = train_gen.calculate_class_weights()

        print(f"\nClass distribution: {train_gen.get_class_distribution()}")
        print(f"Class weights: {self.class_weights}")

        val_gen = GenreDataGenerator(
            self.val_csv,
            self.audio_dir,
            self.feature_extractor
        )
        val_gen.df = val_gen.df.head(self.sample_size // 4)
        val_gen.label_encoder = train_gen.label_encoder
        val_gen.num_classes = self.num_classes
        val_gen.prepare_data()
        self.X_val, self.y_val, _ = val_gen.generate_dataset()

        test_gen = GenreDataGenerator(
            self.test_csv,
            self.audio_dir,
            self.feature_extractor
        )
        test_gen.df = test_gen.df.head(self.sample_size // 4)
        test_gen.label_encoder = train_gen.label_encoder
        test_gen.num_classes = self.num_classes
        test_gen.prepare_data()
        self.X_test, self.y_test, _ = test_gen.generate_dataset()

        print(f"\nDataset shapes:")
        print(f"Train: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Val: {self.X_val.shape}, {self.y_val.shape}")
        print(f"Test: {self.X_test.shape}, {self.y_test.shape}")

    def build_and_compile_model(self, learning_rate: float = 0.001):
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        num_classes = self.y_train.shape[1]

        print(f"\nBuilding model...")
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")

        self.model = GenreCNNClassifier(
            num_classes=num_classes,
            input_shape=input_shape,
            dropout_rate=0.5
        )

        self.model.build_model()
        self.model.compile_model(
            learning_rate=learning_rate,
            class_weights=self.class_weights
        )

        print("\nModel architecture:")
        print(self.model.get_model_summary())

    def train_model(self, epochs: int = 10, batch_size: int = 16):
        print(f"\nStarting quick test training for {epochs} epochs...")

        history = self.model.train(
            train_data=(self.X_train, self.y_train),
            val_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size
        )

        return history

    def evaluate_and_save_metrics(self, history: dict):
        print("\nEvaluating on test set...")
        test_metrics = self.model.evaluate((self.X_test, self.y_test))
        print(f"Test metrics: {test_metrics}")

        print("\nGenerating predictions...")
        y_pred_probs = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        print("\nGenerating metrics and plots...")
        metrics_tracker = MetricsTracker(self.class_names, self.output_dir)

        metrics_tracker.plot_training_history(history)
        metrics_tracker.plot_confusion_matrix(y_true, y_pred, normalize=True)

        report = metrics_tracker.generate_classification_report(y_true, y_pred)
        f1_scores = metrics_tracker.calculate_f1_scores(y_true, y_pred)

        summary = metrics_tracker.save_training_summary(history, test_metrics, f1_scores)

        print("\n" + "="*70)
        print("Quick Test Results:")
        print("="*70)
        print(f"Best Val Accuracy: {summary['best_val_accuracy']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Macro: {f1_scores['f1_macro']:.4f}")
        print(f"F1 Weighted: {f1_scores['f1_weighted']:.4f}")
        print("="*70)

        return summary

    def run(self, epochs: int = 10, batch_size: int = 16, learning_rate: float = 0.001):
        print("="*70)
        print(" "*20 + "QUICK TEST RUN")
        print("="*70)

        self.prepare_small_datasets()
        self.build_and_compile_model(learning_rate)
        history = self.train_model(epochs, batch_size)
        summary = self.evaluate_and_save_metrics(history)

        print("\nQuick test completed successfully!")
        print("Ready to run full training with all data.")
        print("="*70)

        return summary


if __name__ == "__main__":
    pipeline = QuickTestPipeline(
        train_csv=Path("data/processed/splits/train.csv"),
        val_csv=Path("data/processed/splits/val.csv"),
        test_csv=Path("data/processed/splits/test.csv"),
        audio_dir=Path("data/raw/fma_medium/fma_medium"),
        output_dir=Path("models/genre_classifier_test"),
        sample_size=100
    )

    pipeline.run(epochs=10, batch_size=16, learning_rate=0.001)
