import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Dict

from .base_classifier import BaseClassifier
from .losses import FocalLoss
from .custom_layers import SpecAugment
from .model_blocks import residual_block
from .custom_callbacks import LearningRateFormatter
from .data_augmentation import MixupGenerator, apply_test_time_augmentation


class GenreCNNClassifierV2(BaseClassifier):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, int],
                 dropout_rate: float = 0.5,
                 l2_reg: float = 0.02,
                 use_augmentation: bool = True,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__(num_classes, input_shape)
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_augmentation = use_augmentation
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def build_model(self) -> None:
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        if self.use_augmentation:
            x = SpecAugment(freq_mask_param=20, time_mask_param=40,
                           num_freq_masks=2, num_time_masks=2)(x)
        
        x = self._build_stem(x)
        x = self._build_residual_tower(x)
        x = self._build_head(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='genre_cnn_classifier_v2')

    def _build_stem(self, x):
        x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                         kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        return x

    def _build_residual_tower(self, x):
        filters_list = [64, 128, 256, 256]
        dropout_scales = [0.6, 0.7, 0.8, 0.9]
        
        for filters, scale in zip(filters_list, dropout_scales):
            x = residual_block(x, filters, 
                             dropout_rate=self.dropout_rate * scale, 
                             l2_reg=self.l2_reg)
            if filters != 256 or scale != 0.9:
                x = layers.MaxPooling2D((2, 2))(x)
        return x

    def _build_head(self, x):
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])

        x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        return x

    def compile_model(self, learning_rate: float = 0.001,
                     class_weights: Dict[int, float] = None,
                     use_focal_loss: bool = True) -> None:
        if self.model is None:
            raise ValueError("Model not built yet")

        self.class_weights = class_weights
        
        loss = FocalLoss(gamma=self.focal_gamma, alpha=0.25,
                        label_smoothing=self.label_smoothing) if use_focal_loss else \
               keras.losses.CategoricalCrossentropy(label_smoothing=self.label_smoothing)

        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate,
                                            weight_decay=0.01, clipnorm=1.0),
            loss=loss,
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc'),
                     keras.metrics.F1Score(name='f1_score', average='macro')]
        )

    def _create_callbacks(self, X_train, batch_size, epochs):
        initial_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
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

    def train(self, train_data, val_data, epochs: int = 50,
              batch_size: int = 32, use_mixup: bool = True) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")

        X_train, y_train = train_data
        X_val, y_val = val_data
        callbacks = self._create_callbacks(X_train, batch_size, epochs)
        
        if use_mixup:
            mixup_gen = MixupGenerator(X_train, y_train, batch_size)
            train_dataset = mixup_gen.create_dataset()
            steps_per_epoch = len(X_train) // batch_size
            
            history = self.model.fit(
                train_dataset, 
                steps_per_epoch=steps_per_epoch,
                validation_data=(X_val, y_val), 
                epochs=epochs,
                class_weight=self.class_weights, 
                callbacks=callbacks, 
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train, 
                validation_data=(X_val, y_val),
                epochs=epochs, 
                batch_size=batch_size,
                class_weight=self.class_weights, 
                callbacks=callbacks, 
                verbose=1
            )

        return history.history

    def predict(self, features: np.ndarray, use_tta: bool = False) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built yet")
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        if use_tta:
            return apply_test_time_augmentation(self.model, features)
        
        return self.model.predict(features)

    def evaluate(self, test_data) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")
        X_test, y_test = test_data
        return self.model.evaluate(X_test, y_test, return_dict=True, verbose=0)

