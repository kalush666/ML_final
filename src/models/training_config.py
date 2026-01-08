import tensorflow as tf
from tensorflow import keras

from .custom_callbacks import LearningRateFormatter


class TrainingConfig:
    @staticmethod
    def create_callbacks(model, X_train, batch_size: int, epochs: int):
        initial_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        cosine_decay = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=epochs * (len(X_train) // batch_size),
            alpha=0.1
        )

        def lr_schedule(epoch):
            step = epoch * (len(X_train) // batch_size)
            return float(cosine_decay(step).numpy())

        return [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                         restore_best_weights=True, verbose=1),
            keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
            keras.callbacks.ModelCheckpoint('models/checkpoints/genre_best_v2.keras',
                                           monitor='val_f1_score', mode='max',
                                           save_best_only=True, verbose=1),
            LearningRateFormatter()
        ]

    @staticmethod
    def get_optimizer(learning_rate: float):
        return keras.optimizers.AdamW(learning_rate=learning_rate,
                                     weight_decay=0.01, clipnorm=1.0)

    @staticmethod
    def get_metrics():
        return [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.F1Score(name='f1_score', average='macro')
        ]
