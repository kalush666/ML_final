import sys
import os
import multiprocessing
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from training.author.data_loader import FMAAuthorDataLoader
from models.rnn_author_classifier import RNNAuthorClassifier
from models.training_utils.custom_callbacks import LearningRateFormatter


def configure_tensorflow():
    num_cores = multiprocessing.cpu_count()

    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)

    print(f"\nSystem Configuration:")
    print(f"  CPU cores: {num_cores}")
    print(f"  TensorFlow threading: {num_cores} threads")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPUs available: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print(f"  GPUs available: None (using CPU)")


class F1EarlyStopping(keras.callbacks.Callback):

    def __init__(self, monitor='val_f1_score', patience=20, min_delta=0.001, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best = -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best = -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current > self.best + self.min_delta:
            self.best = current
            self.best_epoch = epoch + 1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.verbose:
                print(f"  {self.monitor} did not improve from {self.best:.5f}")

            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                if self.verbose:
                    print(f"Early stopping. Restoring weights from epoch {self.best_epoch}")


class WarmupCosineDecay(keras.callbacks.Callback):

    def __init__(self, initial_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        self.model.optimizer.learning_rate.assign(lr)


def create_callbacks(output_dir, epochs, initial_lr):
    checkpoint_path = str(output_dir / 'checkpoint_best.keras')

    return [
        WarmupCosineDecay(
            initial_lr=initial_lr,
            warmup_epochs=5,
            total_epochs=epochs,
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        F1EarlyStopping(
            monitor='val_f1_score',
            patience=20,
            min_delta=0.001,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        LearningRateFormatter()
    ]


def train_author_classifier():
    print("=" * 60)
    print("RNN Author Classifier Training (Improved)")
    print("=" * 60)

    configure_tensorflow()

    output_dir = Path('models/author_classifier_rnn')
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = 150
    batch_size = 32
    initial_lr = 0.0005

    data_loader = FMAAuthorDataLoader(
        min_tracks_per_artist=10,
        max_artists=50
    )

    train_data, val_data, test_data = data_loader.prepare_dataset(
        test_size=0.2,
        val_size=0.1,
        n_segments=5
    )

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    num_classes = data_loader.get_num_classes()
    input_shape = X_train.shape[1:]

    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial learning rate: {initial_lr}")
    print(f"  Warmup epochs: 5")
    print(f"  Label smoothing: 0.1")

    classifier = RNNAuthorClassifier(
        num_classes=num_classes,
        input_shape=input_shape,
        lstm_units=[128, 64],
        dropout_rate=0.3,
        recurrent_dropout=0.1,
        l2_reg=0.0001,
        use_attention=True,
        use_multihead_attention=True,
        num_attention_heads=4,
        label_smoothing=0.1
    )

    classifier.build_model()
    classifier.compile_model(learning_rate=initial_lr)

    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    _ = classifier.model(dummy_input, training=False)

    checkpoint_path = output_dir / 'checkpoint_best.keras'
    if checkpoint_path.exists():
        print(f"\nFound checkpoint: {checkpoint_path}")
        try:
            classifier.model.load_weights(str(checkpoint_path))
            print("  Resuming from checkpoint weights")
        except Exception as e:
            print(f"  Could not load checkpoint (architecture changed?): {e}")
            print("  Starting fresh training")
    else:
        print("\nNo checkpoint found, starting fresh training")

    print("\nModel Summary:")
    print(classifier.get_model_summary())

    callbacks = create_callbacks(output_dir, epochs, initial_lr)

    print("\nStarting training...")
    history = classifier.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    print("\nEvaluating on test set...")
    results = classifier.evaluate((X_test, y_test))
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")
    print(f"Test F1 Score: {results['f1_score']:.4f}")

    classifier.save_model(output_dir / 'author_classifier_final.keras')

    artist_mapping = {
        'artist_to_idx': data_loader.artist_to_idx,
        'idx_to_artist': data_loader.idx_to_artist
    }
    np.save(output_dir / 'artist_mapping.npy', artist_mapping)

    print(f"\nModel saved to {output_dir}")
    print("Training complete!")

    return classifier, history, results


if __name__ == '__main__':
    train_author_classifier()
