import sys
from pathlib import Path
sys.path.append('src')

import numpy as np
from collections import Counter
from sklearn.utils import resample
from utils.audio_features import AudioFeatureExtractor
from training.data_generator import GenreDataGenerator
from models.genre_classifier_v2 import GenreCNNClassifierV2
from training.metrics import MetricsTracker


class ImprovedGenreTrainingPipeline:
    def __init__(self,
                 train_csv: Path,
                 val_csv: Path,
                 test_csv: Path,
                 audio_dir: Path,
                 output_dir: Path,
                 top_n_classes: int = None):
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_n_classes = top_n_classes
        self.feature_extractor = AudioFeatureExtractor()

    def oversample_minority_classes(self, X: np.ndarray, y: np.ndarray, 
                                     target_samples: int = None) -> tuple:
        print("\nApplying oversampling to balance classes...")
        y_indices = np.argmax(y, axis=1)
        class_counts = Counter(y_indices)
        print(f"Before oversampling: {dict(class_counts)}")
        
        if target_samples is None:
            target_samples = int(np.median(list(class_counts.values())))
        print(f"Target samples per class: {target_samples}")
        
        X_resampled_list = []
        y_resampled_list = []
        
        for class_idx in range(y.shape[1]):
            class_mask = y_indices == class_idx
            X_class = X[class_mask]
            y_class = y[class_mask]
            current_count = len(X_class)
            
            if current_count < target_samples:
                X_oversampled, y_oversampled = resample(
                    X_class, y_class, replace=True,
                    n_samples=target_samples, random_state=42)
                X_resampled_list.append(X_oversampled)
                y_resampled_list.append(y_oversampled)
            else:
                if current_count > target_samples * 2:
                    X_undersampled, y_undersampled = resample(
                        X_class, y_class, replace=False,
                        n_samples=target_samples * 2, random_state=42)
                    X_resampled_list.append(X_undersampled)
                    y_resampled_list.append(y_undersampled)
                else:
                    X_resampled_list.append(X_class)
                    y_resampled_list.append(y_class)
        
        X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.vstack(y_resampled_list)
        
        shuffle_idx = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_idx]
        y_resampled = y_resampled[shuffle_idx]
        
        new_counts = Counter(np.argmax(y_resampled, axis=1))
        print(f"After oversampling: {dict(new_counts)}")
        print(f"Total samples: {len(X_resampled)} (was {len(X)})")
        
        return X_resampled, y_resampled

    def filter_top_classes(self, X: np.ndarray, y: np.ndarray, 
                           class_names: list, n_classes: int) -> tuple:
        print(f"\nFiltering to top {n_classes} classes...")
        y_indices = np.argmax(y, axis=1)
        class_counts = Counter(y_indices)
        
        top_classes = [idx for idx, _ in class_counts.most_common(n_classes)]
        top_class_names = [class_names[idx] for idx in top_classes]
        print(f"Keeping classes: {top_class_names}")
        
        mask = np.isin(y_indices, top_classes)
        X_filtered = X[mask]
        y_filtered_indices = y_indices[mask]
        
        new_label_map = {old: new for new, old in enumerate(top_classes)}
        y_new_indices = np.array([new_label_map[idx] for idx in y_filtered_indices])
        
        from tensorflow import keras
        y_filtered = keras.utils.to_categorical(y_new_indices, num_classes=n_classes)
        print(f"Filtered from {len(X)} to {len(X_filtered)} samples")
        
        return X_filtered, y_filtered, top_class_names

    def prepare_datasets(self, use_oversampling: bool = True, 
                         target_samples_per_class: int = None,
                         n_segments_train: int = 3):
        print(f"Loading training data with {n_segments_train} segments per track...")
        train_gen = GenreDataGenerator(
            self.train_csv, self.audio_dir, self.feature_extractor,
            n_segments=n_segments_train)
        train_gen.prepare_data()
        self.X_train, self.y_train, self.class_names = train_gen.generate_dataset()
        
        if self.top_n_classes:
            self.X_train, self.y_train, self.class_names = self.filter_top_classes(
                self.X_train, self.y_train, self.class_names, self.top_n_classes)
        
        self.num_classes = len(self.class_names)
        
        if use_oversampling:
            self.X_train, self.y_train = self.oversample_minority_classes(
                self.X_train, self.y_train, target_samples_per_class)
        
        y_indices = np.argmax(self.y_train, axis=1)
        class_counts = Counter(y_indices)
        total = len(y_indices)
        self.class_weights = {
            idx: np.sqrt(total / (self.num_classes * count))
            for idx, count in class_counts.items()
        }
        
        print(f"\nClass distribution after processing:")
        for idx, name in enumerate(self.class_names):
            count = class_counts.get(idx, 0)
            weight = self.class_weights.get(idx, 0)
            print(f"  {name}: {count} samples, weight={weight:.2f}")

        print("\nLoading validation data...")
        val_gen = GenreDataGenerator(
            self.val_csv, self.audio_dir, self.feature_extractor)
        val_gen.label_encoder = train_gen.label_encoder
        val_gen.num_classes = train_gen.num_classes
        val_gen.prepare_data()
        self.X_val, self.y_val, _ = val_gen.generate_dataset()
        
        if self.top_n_classes:
            self.X_val, self.y_val, _ = self.filter_top_classes(
                self.X_val, self.y_val, 
                list(train_gen.label_encoder.classes_), 
                self.top_n_classes)

        print("\nLoading test data...")
        test_gen = GenreDataGenerator(
            self.test_csv, self.audio_dir, self.feature_extractor)
        test_gen.label_encoder = train_gen.label_encoder
        test_gen.num_classes = train_gen.num_classes
        test_gen.prepare_data()
        self.X_test, self.y_test, _ = test_gen.generate_dataset()
        
        if self.top_n_classes:
            self.X_test, self.y_test, _ = self.filter_top_classes(
                self.X_test, self.y_test,
                list(train_gen.label_encoder.classes_),
                self.top_n_classes)

        print(f"\nDataset shapes after processing:")
        print(f"Train: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Val: {self.X_val.shape}, {self.y_val.shape}")
        print(f"Test: {self.X_test.shape}, {self.y_test.shape}")

    def build_and_compile_model(self, learning_rate: float = 0.0005,
                                use_focal_loss: bool = True):
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        num_classes = self.y_train.shape[1]

        print(f"\nBuilding improved model (V2)...")
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")

        self.model = GenreCNNClassifierV2(
            num_classes=num_classes, input_shape=input_shape,
            dropout_rate=0.6, l2_reg=0.0005, use_augmentation=True,
            focal_gamma=3.0, label_smoothing=0.1)

        self.model.build_model()
        self.model.compile_model(
            learning_rate=learning_rate,
            class_weights=self.class_weights,
            use_focal_loss=use_focal_loss)

        print("\nModel architecture:")
        print(self.model.get_model_summary())

    def train_model(self, epochs: int = 60, batch_size: int = 32, 
                    use_mixup: bool = True):
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Using mixup: {use_mixup}")

        history = self.model.train(
            train_data=(self.X_train, self.y_train),
            val_data=(self.X_val, self.y_val),
            epochs=epochs, batch_size=batch_size, use_mixup=use_mixup)

        return history

    def evaluate_and_save_metrics(self, history: dict, use_tta: bool = True):
        print("\nEvaluating on test set...")
        test_metrics = self.model.evaluate((self.X_test, self.y_test))
        print(f"Test metrics: {test_metrics}")

        print("\nGenerating predictions (TTA enabled: {})...".format(use_tta))
        y_pred_probs = self.model.predict(self.X_test, use_tta=use_tta)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        print("\nGenerating metrics and plots...")
        metrics_tracker = MetricsTracker(self.class_names, self.output_dir)

        metrics_tracker.plot_training_history(history)
        metrics_tracker.plot_confusion_matrix(y_true, y_pred, normalize=True)
        metrics_tracker.plot_confusion_matrix(y_true, y_pred, normalize=False)

        report = metrics_tracker.generate_classification_report(y_true, y_pred)
        f1_scores = metrics_tracker.calculate_f1_scores(y_true, y_pred)

        summary = metrics_tracker.save_training_summary(history, test_metrics, f1_scores)

        print("\nTraining Summary:")
        print(f"Best Val Accuracy: {summary['best_val_accuracy']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Macro: {f1_scores['f1_macro']:.4f}")
        print(f"F1 Weighted: {f1_scores['f1_weighted']:.4f}")

        return summary

    def save_model(self):
        model_path = self.output_dir / 'genre_classifier_v2_final.keras'
        self.model.save_model(model_path)
        print(f"\nModel saved to: {model_path}")

    def run(self, epochs: int = 60, batch_size: int = 32, 
            learning_rate: float = 0.0005,
            use_oversampling: bool = True,
            use_mixup: bool = True,
            use_focal_loss: bool = True,
            n_segments_train: int = 3):
        print("="*70)
        print(" "*15 + "IMPROVED GENRE CLASSIFIER TRAINING (V2)")
        print("="*70)

        self.prepare_datasets(use_oversampling=use_oversampling, 
                             n_segments_train=n_segments_train)
        self.build_and_compile_model(learning_rate=learning_rate, use_focal_loss=use_focal_loss)
        history = self.train_model(epochs=epochs, batch_size=batch_size, use_mixup=use_mixup)
        self.evaluate_and_save_metrics(history)
        self.save_model()

        print("\n" + "="*70)
        print(" "*20 + "TRAINING COMPLETE!")
        print("="*70)


if __name__ == '__main__':
    train_csv = Path('data/processed/splits/train.csv')
    val_csv = Path('data/processed/splits/val.csv')
    test_csv = Path('data/processed/splits/test.csv')
    audio_dir = Path('data/raw/fma_medium/fma_medium')
    output_dir = Path('models/genre_classifier_v2')

    pipeline = ImprovedGenreTrainingPipeline(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        audio_dir=audio_dir,
        output_dir=output_dir,
        top_n_classes=8
    )

    pipeline.run(
        epochs=60,
        batch_size=32,
        learning_rate=0.0003,
        use_oversampling=True,
        use_mixup=True,
        use_focal_loss=True, 
        n_segments_train=4
    )
