from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

from .base_classifier import BaseClassifier


@register_keras_serializable(package="AuthorClassifier")
class AttentionPooling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = layers.Dense(1, activation='tanh')
        super().build(input_shape)

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        weighted_sum = tf.reduce_sum(inputs * attention_weights, axis=1)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="AuthorClassifier")
class ResidualBiLSTMBlock(layers.Layer):

    def __init__(
        self,
        units: int,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.1,
        l2_reg: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.bilstm = layers.Bidirectional(
            layers.LSTM(
                self.units,
                return_sequences=True,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            )
        )
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)

        input_dim = input_shape[-1]
        output_dim = self.units * 2

        if input_dim != output_dim:
            self.projection = layers.Dense(output_dim, use_bias=False)
        else:
            self.projection = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.bilstm(inputs)
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)

        if self.projection is not None:
            residual = self.projection(inputs)
        else:
            residual = inputs

        return x + residual

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units * 2)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout': self.recurrent_dropout,
            'l2_reg': self.l2_reg
        })
        return config


class RNNAuthorClassifier(BaseClassifier):

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, ...],
        lstm_units: List[int] = None,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.1,
        l2_reg: float = 0.0001,
        use_attention: bool = True,
        use_multihead_attention: bool = True,
        num_attention_heads: int = 4,
        label_smoothing: float = 0.1
    ):
        super().__init__(num_classes, input_shape)

        self.lstm_units = lstm_units if lstm_units is not None else [128, 64]
        self.use_attention = use_attention
        self.use_multihead_attention = use_multihead_attention
        self.num_attention_heads = num_attention_heads

        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.label_smoothing = label_smoothing

    def build_model(self) -> None:
        inputs = layers.Input(shape=self.input_shape)

        x = self._build_input_block(inputs)
        x = self._build_recurrent_blocks(x)

        if self.use_multihead_attention:
            x = self._build_multihead_attention(x)

        x = self._build_pooling(x)
        outputs = self._build_classification_head(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def _build_input_block(self, inputs):
        x = layers.Dense(
            self.lstm_units[0] * 2,
            activation='relu',
            name='input_projection'
        )(inputs)
        x = layers.GaussianNoise(0.02)(x)
        x = layers.LayerNormalization(name='input_norm')(x)
        return x

    def _build_recurrent_blocks(self, x):
        for i, units in enumerate(self.lstm_units):
            x = ResidualBiLSTMBlock(
                units=units,
                dropout_rate=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                l2_reg=self.l2_reg,
                name=f'res_bilstm_{i}'
            )(x)
        return x

    def _build_multihead_attention(self, x):
        key_dim = self.lstm_units[-1] * 2 // self.num_attention_heads

        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name='multihead_attention'
        )(x, x)

        x = layers.Add(name='attention_residual')([x, attention_output])
        x = layers.LayerNormalization(name='attention_norm')(x)

        return x

    def _build_pooling(self, x):
        if self.use_attention:
            x = AttentionPooling(name='attention_pooling')(x)
        else:
            x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        return x

    def _build_classification_head(self, x):
        x = layers.Dense(
            256,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_1'
        )(x)
        x = layers.LayerNormalization(name='dense_1_norm')(x)
        x = layers.ReLU(name='dense_1_relu')(x)
        x = layers.Dropout(self.dropout_rate * 0.75, name='dense_1_dropout')(x)

        x = layers.Dense(
            128,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_2'
        )(x)
        x = layers.LayerNormalization(name='dense_2_norm')(x)
        x = layers.ReLU(name='dense_2_relu')(x)
        x = layers.Dropout(self.dropout_rate * 0.75, name='dense_2_dropout')(x)

        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        return outputs

    def compile_model(self, learning_rate: float = 0.001) -> None:
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")

        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            clipnorm=1.0
        )

        loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=self.label_smoothing
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.F1Score(name='f1_score', average='macro')
            ]
        )

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        epochs: int,
        batch_size: int,
        callbacks: Optional[List] = None
    ) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")

        X_train, y_train = train_data
        X_val, y_val = val_data

        if callbacks is None:
            callbacks = []

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history.history

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built yet.")
        return self.model.predict(features, verbose=0)

    def predict_classes(self, features: np.ndarray) -> np.ndarray:
        probs = self.predict(features)
        return np.argmax(probs, axis=1)

    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet.")

        X_test, y_test = test_data
        results = self.model.evaluate(X_test, y_test, verbose=0)

        metric_names = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1_score']
        return {name: value for name, value in zip(metric_names, results)}
