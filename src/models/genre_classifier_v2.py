import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Dict, List, Optional

from .base_classifier import BaseClassifier
from .training_utils.losses import FocalLoss, AdaptiveFocalLoss, create_gtzan_adaptive_loss
from .layers.custom_layers import SpecAugment
from .architecture.architecture_builder import ArchitectureBuilder
from .training_utils.training_config import TrainingConfig
from .inference.inference_engine import InferenceEngine
from .training_utils.data_augmentation import MixupGenerator


class GenreCNNClassifierV2(BaseClassifier):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, int],
                 dropout_rate: float = 0.5,
                 l2_reg: float = 0.0001,
                 use_augmentation: bool = True,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 use_adaptive_loss: bool = False,
                 per_class_gamma: Optional[List[float]] = None,
                 confidence_penalty: float = 0.15):
        super().__init__(num_classes, input_shape)
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_augmentation = use_augmentation
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_adaptive_loss = use_adaptive_loss
        self.per_class_gamma = per_class_gamma
        self.confidence_penalty = confidence_penalty

    def build_model(self) -> None:
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        if self.use_augmentation:
            x = SpecAugment(freq_mask_param=20, time_mask_param=40,
                           num_freq_masks=2, num_time_masks=2)(x)
        
        x = ArchitectureBuilder.build_stem(x, self.l2_reg)
        x = ArchitectureBuilder.build_residual_tower(x, self.dropout_rate, self.l2_reg)
        x = ArchitectureBuilder.build_head(x, self.dropout_rate, self.l2_reg)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs, name='genre_cnn_classifier_v2')

    def compile_model(self, learning_rate: float = 0.001,
                     class_weights: Dict[int, float] = None,
                     use_focal_loss: bool = True) -> None:
        if self.model is None:
            raise ValueError("Model not built yet")

        self.class_weights = class_weights
        
        if self.use_adaptive_loss:
            loss = create_gtzan_adaptive_loss(label_smoothing=self.label_smoothing)
            print("Using Adaptive Focal Loss with per-class gamma and confusion penalties")
        elif use_focal_loss:
            loss = FocalLoss(gamma=self.focal_gamma,
                            label_smoothing=self.label_smoothing)
            print(f"Using Standard Focal Loss (gamma={self.focal_gamma})")
        else:
            loss = keras.losses.CategoricalCrossentropy(label_smoothing=self.label_smoothing)
            print("Using Categorical Cross-Entropy")

        self.model.compile(
            optimizer=TrainingConfig.get_optimizer(learning_rate),
            loss=loss,
            metrics=TrainingConfig.get_metrics()
        )

    def train(self, train_data, val_data, epochs: int = 50,
              batch_size: int = 32, use_mixup: bool = True) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")

        X_train, y_train = train_data
        X_val, y_val = val_data
        callbacks = TrainingConfig.create_callbacks(self.model, X_train, batch_size, epochs)
        
        if use_mixup:
            mixup_gen = MixupGenerator(X_train, y_train, batch_size)
            train_dataset = mixup_gen.create_dataset()
            steps_per_epoch = len(X_train) // batch_size
            
            history = self.model.fit(
                train_dataset, 
                steps_per_epoch=steps_per_epoch,
                validation_data=(X_val, y_val), 
                epochs=epochs,
                class_weight=self.class_weights, 
                callbacks=callbacks, 
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train, 
                validation_data=(X_val, y_val),
                epochs=epochs, 
                batch_size=batch_size,
                class_weight=self.class_weights, 
                callbacks=callbacks, 
                verbose=1
            )

        return history.history

    def predict(self, features: np.ndarray, use_tta: bool = False) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built yet")
        return InferenceEngine.predict(self.model, features, use_tta)

    def evaluate(self, test_data) -> Dict:
        if self.model is None:
            raise ValueError("Model not built yet")
        return InferenceEngine.evaluate(self.model, test_data)

