import numpy as np
from typing import Dict, List

from ..training_utils.data_augmentation import apply_test_time_augmentation


class InferenceEngine:
    @staticmethod
    def predict(model, features: np.ndarray, use_tta: bool = False) -> np.ndarray:
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        if use_tta:
            return apply_test_time_augmentation(model, features)
        
        return model.predict(features)

    @staticmethod
    def predict_multi_segment(model, segments: List[np.ndarray], 
                              aggregation: str = 'mean') -> np.ndarray:
        predictions = []
        for segment in segments:
            if len(segment.shape) == 2:
                segment = np.expand_dims(segment, axis=0)
            pred = model.predict(segment, verbose=0)
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        if aggregation == 'mean':
            return np.mean(predictions, axis=0, keepdims=True)
        elif aggregation == 'max':
            return np.max(predictions, axis=0, keepdims=True)
        elif aggregation == 'vote':
            votes = np.argmax(predictions, axis=1)
            winner = np.bincount(votes).argmax()
            result = np.zeros((1, predictions.shape[1]))
            result[0, winner] = 1.0
            return result
        else:
            return np.mean(predictions, axis=0, keepdims=True)

    @staticmethod
    def evaluate(model, test_data) -> Dict:
        X_test, y_test = test_data
        return model.evaluate(X_test, y_test, return_dict=True, verbose=0)
