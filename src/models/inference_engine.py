import numpy as np
from typing import Dict

from .data_augmentation import apply_test_time_augmentation


class InferenceEngine:
    @staticmethod
    def predict(model, features: np.ndarray, use_tta: bool = False) -> np.ndarray:
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        if use_tta:
            return apply_test_time_augmentation(model, features)
        
        return model.predict(features)

    @staticmethod
    def evaluate(model, test_data) -> Dict:
        X_test, y_test = test_data
        return model.evaluate(X_test, y_test, return_dict=True, verbose=0)
