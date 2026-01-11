import tensorflow as tf
from tensorflow import keras
import numpy as np

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    try:
        register_serializable = keras.utils.register_keras_serializable
    except AttributeError:
        def register_serializable(package=None, name=None):
            def decorator(cls):
                return cls
            return decorator if package is None and name is None else decorator


@register_serializable()
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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing
        })
        return config


@register_serializable()
class AdaptiveFocalLoss(keras.losses.Loss):
    
    def __init__(self, 
                 base_gamma: float = 2.0,
                 per_class_gamma: list = None,
                 label_smoothing: float = 0.1,
                 confidence_penalty: float = 0.1,
                 confusion_matrix_weights: list = None,
                 num_classes: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_gamma = base_gamma
        self.label_smoothing = label_smoothing
        self.confidence_penalty = confidence_penalty
        self.num_classes = num_classes
        
        if per_class_gamma is not None:
            self.per_class_gamma = tf.constant(per_class_gamma, dtype=tf.float32)
        else:
            self.per_class_gamma = tf.constant([base_gamma] * num_classes, dtype=tf.float32)
        
        if confusion_matrix_weights is not None:
            self.confusion_weights = tf.constant(confusion_matrix_weights, dtype=tf.float32)
        else:
            self.confusion_weights = None
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true_smooth = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        else:
            y_true_smooth = y_true
        
        cross_entropy = keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
        true_class_indices = tf.argmax(y_true, axis=-1)
        sample_gamma = tf.gather(self.per_class_gamma, true_class_indices)
        
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1 - p_t, sample_gamma)
        
        focal_loss = focal_weight * cross_entropy
        
        if self.confidence_penalty > 0:
            pred_class_indices = tf.argmax(y_pred, axis=-1)
            is_wrong = tf.cast(tf.not_equal(true_class_indices, pred_class_indices), tf.float32)
            max_confidence = tf.reduce_max(y_pred, axis=-1)
            confidence_loss = is_wrong * max_confidence * self.confidence_penalty
            focal_loss = focal_loss + confidence_loss
        
        if self.confusion_weights is not None:
            pred_class_indices = tf.argmax(y_pred, axis=-1)
            indices = tf.stack([tf.cast(true_class_indices, tf.int32), 
                               tf.cast(pred_class_indices, tf.int32)], axis=1)
            confusion_penalty = tf.gather_nd(self.confusion_weights, indices)
            focal_loss = focal_loss * (1.0 + confusion_penalty)
        
        return focal_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'base_gamma': self.base_gamma,
            'per_class_gamma': self.per_class_gamma.numpy().tolist() if hasattr(self.per_class_gamma, 'numpy') else list(self.per_class_gamma),
            'label_smoothing': self.label_smoothing,
            'confidence_penalty': self.confidence_penalty,
            'num_classes': self.num_classes,
            'confusion_matrix_weights': self.confusion_weights.numpy().tolist() if self.confusion_weights is not None else None
        })
        return config


def create_gtzan_adaptive_loss(label_smoothing: float = 0.1) -> AdaptiveFocalLoss:
    per_class_gamma = [
        2.0, 2.0, 2.5, 3.5, 2.5, 2.0, 2.5, 3.5, 2.5, 4.0
    ]
    
    num_classes = 10
    confusion_weights = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    confusion_weights[9, 0] = 0.5
    confusion_weights[3, 7] = 0.3
    confusion_weights[7, 3] = 0.3
    confusion_weights[9, 6] = 0.2
    confusion_weights[9, 2] = 0.2
    
    return AdaptiveFocalLoss(
        base_gamma=2.5,
        per_class_gamma=per_class_gamma,
        label_smoothing=label_smoothing,
        confidence_penalty=0.15,
        confusion_matrix_weights=confusion_weights.tolist(),
        num_classes=num_classes
    )

