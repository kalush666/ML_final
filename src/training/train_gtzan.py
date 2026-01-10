"""
GTZAN Genre Classifier Training
================================
Trains a CNN on the GTZAN dataset for music genre classification.
Expected accuracy: 80-85%
"""

import sys
from pathlib import Path
sys.path.append('src')

import numpy as np
import pandas as pd
from collections import Counter
from tensorflow import keras

from utils.audio_features import AudioFeatureExtractor
from models.genre_classifier_v2 import GenreCNNClassifierV2
from training.metrics import MetricsTracker


GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


class GTZANDataGenerator:
    """Data generator for GTZAN dataset."""
    
    def __init__(self, csv_path: Path, feature_extractor: AudioFeatureExtractor,
                 n_segments: int = 1):
        self.csv_path = Path(csv_path)
        self.feature_extractor = feature_extractor
        self.n_segments = n_segments
        self.df = pd.read_csv(csv_path)
        
    def generate_dataset(self):
        """Load audio and extract features."""
        features_list = []
        labels_list = []
        
        genre_to_idx = {g: i for i, g in enumerate(GTZAN_GENRES)}
        
        total = len(self.df)
        for idx, row in self.df.iterrows():
            audio_path = Path(row['path'])
            genre = row['genre']
            
            if genre not in genre_to_idx:
                continue
                
            try:
                if self.n_segments > 1:
                    segments = self.feature_extractor.extract_multi_segments(
                        audio_path, n_segments=self.n_segments, random_offset=True)
                    if segments is not None:
                        for seg in segments:
                            features_list.append(seg)
                            labels_list.append(genre_to_idx[genre])
                else:
                    audio = self.feature_extractor.load_audio(audio_path)
                    features = self.feature_extractor.extract_combined_features(audio)
                    features_list.append(features)
                    labels_list.append(genre_to_idx[genre])
                    
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total} tracks...")
        
        X = np.array(features_list)
        y = keras.utils.to_categorical(labels_list, num_classes=len(GTZAN_GENRES))
        
        return X, y, GTZAN_GENRES


class GTZANTrainingPipeline:
    """Training pipeline for GTZAN dataset."""
    
    def __init__(self, 
                 train_csv: Path,
                 val_csv: Path,
                 test_csv: Path,
                 output_dir: Path):
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = AudioFeatureExtractor()
        self.num_classes = len(GTZAN_GENRES)
        self.class_names = GTZAN_GENRES
        
    def prepare_datasets(self, n_segments_train: int = 3):
        """Load and prepare all datasets."""
        print("\nLoading training data...")
        train_gen = GTZANDataGenerator(
            self.train_csv, self.feature_extractor, n_segments=n_segments_train)
        self.X_train, self.y_train, _ = train_gen.generate_dataset()
        print(f"  Training samples: {len(self.X_train)}")
        
        print("\nLoading validation data...")
        val_gen = GTZANDataGenerator(
            self.val_csv, self.feature_extractor, n_segments=1)
        self.X_val, self.y_val, _ = val_gen.generate_dataset()
        print(f"  Validation samples: {len(self.X_val)}")
        
        print("\nLoading test data...")
        test_gen = GTZANDataGenerator(
            self.test_csv, self.feature_extractor, n_segments=1)
        self.X_test, self.y_test, _ = test_gen.generate_dataset()
        print(f"  Test samples: {len(self.X_test)}")
        
        # Compute class weights (sqrt scaling)
        y_indices = np.argmax(self.y_train, axis=1)
        class_counts = Counter(y_indices)
        total = len(y_indices)
        self.class_weights = {
            idx: np.sqrt(total / (self.num_classes * count))
            for idx, count in class_counts.items()
        }
        
        print("\nClass distribution (training):")
        for idx, name in enumerate(self.class_names):
            count = class_counts.get(idx, 0)
            weight = self.class_weights.get(idx, 1.0)
            print(f"  {name}: {count} samples, weight={weight:.2f}")
            
    def build_model(self, learning_rate: float = 0.0003):
        """Build and compile the model."""
        print("\nBuilding model...")
        
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        print(f"  Input shape: {input_shape}")
        print(f"  Number of classes: {self.num_classes}")
        
        self.model = GenreCNNClassifierV2(
            num_classes=self.num_classes,
            input_shape=input_shape,
            dropout_rate=0.5,
            l2_reg=0.0001,
            use_augmentation=True,
            focal_gamma=2.0,
            label_smoothing=0.1
        )
        self.model.build_model()
        self.model.compile_model(
            learning_rate=learning_rate,
            class_weights=self.class_weights,
            use_focal_loss=True
        )
        
        print("\nModel Summary:")
        print(self.model.get_model_summary())
        
    def train(self, epochs: int = 60, batch_size: int = 32, use_mixup: bool = True):
        """Train the model."""
        print(f"\nStarting training for {epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print(f"  Using mixup: {use_mixup}")
        
        history = self.model.train(
            train_data=(self.X_train, self.y_train),
            val_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            use_mixup=use_mixup
        )
        
        return history
    
    def evaluate(self, history):
        """Evaluate on test set and generate reports."""
        print("\nEvaluating on test set...")
        
        test_metrics = self.model.evaluate((self.X_test, self.y_test))
        print(f"Test metrics: {test_metrics}")
        
        # Generate predictions
        y_pred_probs = self.model.predict(self.X_test, use_tta=True)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Generate metrics and plots
        print("\nGenerating metrics and plots...")
        metrics_tracker = MetricsTracker(self.class_names, self.output_dir)
        
        metrics_tracker.plot_training_history(history)
        metrics_tracker.plot_confusion_matrix(y_true, y_pred, normalize=True)
        metrics_tracker.plot_confusion_matrix(y_true, y_pred, normalize=False)
        
        report = metrics_tracker.generate_classification_report(y_true, y_pred)
        f1_scores = metrics_tracker.calculate_f1_scores(y_true, y_pred)
        
        summary = metrics_tracker.save_training_summary(history, test_metrics, f1_scores)
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Best Val Accuracy: {summary['best_val_accuracy']:.4f}")
        print(f"Test Accuracy:     {test_metrics['accuracy']:.4f}")
        print(f"Test Loss:         {test_metrics['loss']:.4f}")
        print(f"F1 Macro:          {f1_scores['f1_macro']:.4f}")
        print(f"F1 Weighted:       {f1_scores['f1_weighted']:.4f}")
        print("="*50)
            
        return test_metrics, f1_scores
    
    def save_model(self):
        """Save the trained model."""
        model_path = self.output_dir / 'gtzan_classifier_final.keras'
        self.model.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
        
    def run(self, epochs: int = 60, batch_size: int = 32, 
            learning_rate: float = 0.0003, n_segments: int = 3,
            use_mixup: bool = True):
        """Run full training pipeline."""
        print("="*70)
        print(" "*15 + "GTZAN GENRE CLASSIFIER TRAINING")
        print("="*70)
        print(f"\nDataset: GTZAN (10 genres, 1000 tracks)")
        print(f"Expected accuracy: 80-85%")
        
        self.prepare_datasets(n_segments_train=n_segments)
        self.build_model(learning_rate=learning_rate)
        history = self.train(epochs=epochs, batch_size=batch_size, use_mixup=use_mixup)
        self.evaluate(history)
        self.save_model()
        
        print("\n" + "="*70)
        print(" "*20 + "TRAINING COMPLETE!")
        print("="*70)


if __name__ == '__main__':
    # Paths
    train_csv = Path('data/processed/gtzan_splits/train.csv')
    val_csv = Path('data/processed/gtzan_splits/val.csv')
    test_csv = Path('data/processed/gtzan_splits/test.csv')
    output_dir = Path('models/gtzan_classifier')
    
    # Check if splits exist
    if not train_csv.exists():
        print("GTZAN splits not found!")
        print("Run first: python src/data/download_gtzan.py")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = GTZANTrainingPipeline(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_dir=output_dir
    )
    
    pipeline.run(
        epochs=60,
        batch_size=32,
        learning_rate=0.0003,
        n_segments=3,
        use_mixup=True
    )
