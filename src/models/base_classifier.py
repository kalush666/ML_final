from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Tuple, Dict


class BaseClassifier(ABC):

    def __init__(self, num_classes: int, input_shape: Tuple[int, ...]):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    @abstractmethod
    def build_model(self) -> None:
        pass

    @abstractmethod
    def compile_model(self, learning_rate: float = 0.001) -> None:
        pass

    @abstractmethod
    def train(self,
              train_data,
              val_data,
              epochs: int,
              batch_size: int) -> Dict:
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, test_data) -> Dict:
        pass

    def save_model(self, filepath: Path) -> None:
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save(filepath)

    def load_model(self, filepath: Path) -> None:
        from tensorflow import keras
        self.model = keras.models.load_model(filepath)

    def get_model_summary(self) -> str:
        if self.model is None:
            raise ValueError("Model not built yet")

        summary_str = []
        self.model.summary(print_fn=lambda x: summary_str.append(x))
        return '\n'.join(summary_str)
