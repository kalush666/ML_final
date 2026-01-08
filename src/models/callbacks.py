import tensorflow as tf
from tensorflow import keras


class LearningRateFormatter(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"\nEpoch {epoch + 1} - Learning Rate: {lr:.6f}")
