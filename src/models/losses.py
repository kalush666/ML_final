import tensorflow as tf
from tensorflow import keras


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
        
        # Compute cross-entropy: -y_true * log(y_pred)
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # For focal loss, we weight each class's contribution by (1 - p)^gamma
        # where p is the predicted probability for that class
        focal_weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        
        # Apply focal weight to cross-entropy and sum across classes
        focal_loss = tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
        
        return focal_loss

