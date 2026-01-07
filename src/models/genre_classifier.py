import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Dict
from .base_classifier import BaseClassifier


class GenreCNNClassifier(BaseClassifier):

    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, int],
                 dropout_rate: float = 0.5):
        super().__init__(num_classes, input_shape)
        self.dropout_rate = dropout_rate

    def build_model(self) -> None:
        inputs = keras.Input(shape=self.input_shape)

        x = layers.Reshape((*self.input_shape, 1))(inputs)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.7)(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='genre_cnn_classifier')

    def compile_model(self,
                     learning_rate: float = 0.001,
                     class_weights: Dict[int, float] = None) -> None:
        if self.model is None:
            raise ValueError("Model not built yet")

        self.class_weights = class_weights

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

    def train(self,
              train_data,
              val_data,
              epochs: int = 50,
              batch_size: int = 32) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")

        X_train, y_train = train_data
        X_val, y_val = val_data

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'models/checkpoints/genre_best.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return history.history

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built yet")

        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)

        predictions = self.model.predict(features)
        return predictions

    def evaluate(self, test_data) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")

        X_test, y_test = test_data
        results = self.model.evaluate(X_test, y_test, return_dict=True, verbose=0)
        return results
