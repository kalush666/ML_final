import tensorflow as tf
from tensorflow.keras import layers


class DropPath(layers.Layer):
    def __init__(self, drop_prob: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.floor(random_tensor)
        return x / keep_prob * binary_mask

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


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
        batch_size, time_steps, freq_bins = shape[0], shape[1], shape[2]
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
