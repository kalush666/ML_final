from tensorflow import keras
from tensorflow.keras import layers

from ..layers.custom_layers import DropPath


def squeeze_excitation_block(x, ratio=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def residual_block(x, filters, kernel_size=(3, 3), dropout_rate=0.3, l2_reg=0.02, 
                   use_se=True, drop_path_rate=0.0):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.Conv2D(filters, kernel_size, padding='same',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    
    if use_se:
        x = squeeze_excitation_block(x)

    if drop_path_rate > 0:
        x = DropPath(drop_path_rate)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same',
                                kernel_regularizer=keras.regularizers.l2(l2_reg))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x
