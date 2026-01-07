import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Dict
from .base_classifier import BaseClassifier


class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, 
                 label_smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        focal_loss = focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
        
        return focal_loss


class SpecAugment(layers.Layer):
    def __init__(self, freq_mask_param: int = 10, time_mask_param: int = 20, 
                 num_freq_masks: int = 2, num_time_masks: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        shape = tf.shape(inputs)
        batch_size = shape[0]
        time_steps = shape[1]
        freq_bins = shape[2]
        augmented = inputs
        
        for _ in range(self.num_freq_masks):
            f = tf.random.uniform([], 0, self.freq_mask_param, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, freq_bins - f, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([batch_size, time_steps, f0, 1]),
                tf.zeros([batch_size, time_steps, f, 1]),
                tf.ones([batch_size, time_steps, freq_bins - f0 - f, 1])
            ], axis=2)
            augmented = augmented * mask
        
        for _ in range(self.num_time_masks):
            t = tf.random.uniform([], 0, self.time_mask_param, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, time_steps - t, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([batch_size, t0, freq_bins, 1]),
                tf.zeros([batch_size, t, freq_bins, 1]),
                tf.ones([batch_size, time_steps - t0 - t, freq_bins, 1])
            ], axis=1)
            augmented = augmented * mask
        
        return augmented


def residual_block(x, filters, kernel_size=(3, 3), dropout_rate=0.3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


class LearningRateFormatter(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"\nEpoch {epoch + 1} - Learning Rate: {lr:.6f}")


class GenreCNNClassifierV2(BaseClassifier):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, int],
                 dropout_rate: float = 0.4,
                 use_augmentation: bool = True,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__(num_classes, input_shape)
        self.dropout_rate = dropout_rate
        self.use_augmentation = use_augmentation
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def build_model(self) -> None:
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        if self.use_augmentation:
            x = SpecAugment(freq_mask_param=15, time_mask_param=30,
                           num_freq_masks=2, num_time_masks=2)(x)
        
        x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = residual_block(x, 64, dropout_rate=self.dropout_rate * 0.5)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = residual_block(x, 128, dropout_rate=self.dropout_rate * 0.6)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = residual_block(x, 256, dropout_rate=self.dropout_rate * 0.7)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = residual_block(x, 512, dropout_rate=self.dropout_rate * 0.8)
        
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        
        x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='genre_cnn_classifier_v2')

    def compile_model(self, learning_rate: float = 0.001,
                     class_weights: Dict[int, float] = None,
                     use_focal_loss: bool = True) -> None:
        if self.model is None:
            raise ValueError("Model not built yet")

        self.class_weights = class_weights
        
        if use_focal_loss:
            loss = FocalLoss(gamma=self.focal_gamma, alpha=0.25,
                           label_smoothing=self.label_smoothing)
        else:
            loss = keras.losses.CategoricalCrossentropy(
                label_smoothing=self.label_smoothing)

        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01),
            loss=loss,
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc'),
                     keras.metrics.F1Score(name='f1_score', average='macro')]
        )

    def train(self, train_data, val_data, epochs: int = 50,
              batch_size: int = 32, use_mixup: bool = True) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")

        X_train, y_train = train_data
        X_val, y_val = val_data

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                         restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=5, min_lr=1e-7, verbose=1),
            keras.callbacks.ModelCheckpoint('models/checkpoints/genre_best_v2.keras',
                                           monitor='val_f1_score', mode='max',
                                           save_best_only=True, verbose=1),
            LearningRateFormatter()
        ]
        
        if use_mixup:
            train_dataset = self._create_mixup_dataset(X_train, y_train, batch_size)
            steps_per_epoch = len(X_train) // batch_size
            history = self.model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                                    validation_data=(X_val, y_val), epochs=epochs,
                                    class_weight=self.class_weights, callbacks=callbacks, verbose=1)
        else:
            history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                    epochs=epochs, batch_size=batch_size,
                                    class_weight=self.class_weights, callbacks=callbacks, verbose=1)

        return history.history
    
    def _create_mixup_dataset(self, X, y, batch_size, alpha=0.2):
        def mixup(x1, y1, x2, y2, alpha):
            lam = np.random.beta(alpha, alpha)
            x = lam * x1 + (1 - lam) * x2
            y = lam * y1 + (1 - lam) * y2
            return x, y
        
        def mixup_generator():
            indices = np.arange(len(X))
            while True:
                np.random.shuffle(indices)
                for i in range(0, len(indices) - batch_size, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_x = X[batch_indices]
                    batch_y = y[batch_indices]
                    
                    if np.random.random() > 0.5:
                        shuffle_indices = np.random.permutation(batch_size)
                        mixed_x, mixed_y = [], []
                        for j in range(batch_size):
                            x_mix, y_mix = mixup(batch_x[j], batch_y[j],
                                                batch_x[shuffle_indices[j]],
                                                batch_y[shuffle_indices[j]], alpha)
                            mixed_x.append(x_mix)
                            mixed_y.append(y_mix)
                        yield np.array(mixed_x), np.array(mixed_y)
                    else:
                        yield batch_x, batch_y
        
        output_signature = (
            tf.TensorSpec(shape=(batch_size, *X.shape[1:]), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, y.shape[1]), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(mixup_generator, output_signature=output_signature)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built yet")
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        return self.model.predict(features)

    def evaluate(self, test_data) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")
        X_test, y_test = test_data
        return self.model.evaluate(X_test, y_test, return_dict=True, verbose=0)
