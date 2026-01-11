import tensorflow as tf
from tensorflow import keras


class LearningRateFormatter(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"\nEpoch {epoch + 1} - Learning Rate: {lr:.6f}")


class F1EarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor='val_f1_score', patience=15, min_delta=0.001, verbose=1, restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best = -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best = -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
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
                print(f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f} (last improved at epoch {self.best_epoch})")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                if self.verbose:
                    print(f"Restoring model weights from the end of the best epoch: {self.best_epoch}")
                    print(f"Early stopping at epoch {self.stopped_epoch}")
