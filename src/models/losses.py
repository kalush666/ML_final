import tensorflow as tf
from tensorflow import keras


class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        cross_entropy = keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        return focal_weight * cross_entropy

