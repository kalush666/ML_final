import numpy as np
import tensorflow as tf


class MixupGenerator:
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
    
    def _mixup(self, x1, y1, x2, y2):
        lam = np.random.beta(self.alpha, self.alpha)
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y
    
    def generate(self):
        indices = np.arange(len(self.X))
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices) - self.batch_size, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_x = self.X[batch_indices]
                batch_y = self.y[batch_indices]
                
                if np.random.random() > 0.5:
                    shuffle_indices = np.random.permutation(self.batch_size)
                    mixed_x, mixed_y = [], []
                    for j in range(self.batch_size):
                        x_mix, y_mix = self._mixup(
                            batch_x[j], batch_y[j],
                            batch_x[shuffle_indices[j]],
                            batch_y[shuffle_indices[j]]
                        )
                        mixed_x.append(x_mix)
                        mixed_y.append(y_mix)
                    yield np.array(mixed_x), np.array(mixed_y)
                else:
                    yield batch_x, batch_y
    
    def create_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(self.batch_size, *self.X.shape[1:]), dtype=tf.float32),
            tf.TensorSpec(shape=(self.batch_size, self.y.shape[1]), dtype=tf.float32)
        )
        dataset = tf.data.Dataset.from_generator(
            self.generate,
            output_signature=output_signature
        )
        return dataset.prefetch(tf.data.AUTOTUNE)


def apply_test_time_augmentation(model, features):
    preds = []
    preds.append(model.predict(features, verbose=0))
    preds.append(model.predict(features[:, ::-1, :], verbose=0))
    preds.append(model.predict(features + np.random.normal(0, 0.01, features.shape), verbose=0))
    return np.mean(preds, axis=0)
