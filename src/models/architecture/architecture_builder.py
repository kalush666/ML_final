from tensorflow import keras
from tensorflow.keras import layers

from .model_blocks import residual_block


class ArchitectureBuilder:
    @staticmethod
    def build_stem(x, l2_reg: float):
        x = layers.GaussianNoise(0.1)(x)
        x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                         kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        return x

    @staticmethod
    def build_residual_tower(x, dropout_rate: float, l2_reg: float):
        filters_list = [64, 128, 256, 256]
        dropout_scales = [0.6, 0.7, 0.8, 0.9]
        drop_path_rates = [0.1, 0.15, 0.2, 0.25]
        
        for filters, scale, dp_rate in zip(filters_list, dropout_scales, drop_path_rates):
            x = residual_block(x, filters, 
                             dropout_rate=dropout_rate * scale, 
                             l2_reg=l2_reg,
                             drop_path_rate=dp_rate)
            if filters != 256 or scale != 0.9:
                x = layers.MaxPooling2D((2, 2))(x)
        return x

    @staticmethod
    def build_head(x, dropout_rate: float, l2_reg: float):
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])

        x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        return x
