from pathlib import Path
from models.genre_classifier_v2 import GenreCNNClassifierV2
from .config import GTZANConfig


class GTZANModelTrainer:

    def __init__(self, config: GTZANConfig, pretrained_model_path: Path = None):
        self.config = config
        self.classifier_model = None
        self.pretrained_model_path = pretrained_model_path

    def initialize_model(self, input_shape: tuple, class_weights: dict):
        print("\nBuilding model...")
        print(f"  Input shape: {input_shape}")
        print(f"  Number of classes: {self.config.number_of_classes}")
        print(f"  Adaptive focal loss: {self.config.use_adaptive_focal_loss}")

        self.classifier_model = GenreCNNClassifierV2(
            num_classes=self.config.number_of_classes,
            input_shape=input_shape,
            dropout_rate=self.config.dropout_rate,
            l2_reg=self.config.l2_regularization,
            use_augmentation=self.config.enable_specaugment,
            focal_gamma=self.config.focal_loss_gamma,
            label_smoothing=self.config.label_smoothing_factor,
            use_adaptive_loss=self.config.use_adaptive_focal_loss,
            per_class_gamma=self.config.per_class_gamma,
            confidence_penalty=self.config.confidence_penalty
        )

        self.classifier_model.build_model()

        if self.pretrained_model_path and self.pretrained_model_path.exists():
            print(f"\nLoading pretrained weights from: {self.pretrained_model_path}")
            try:
                from tensorflow import keras
                from models.layers.custom_layers import SpecAugment, DropPath
                from models.training_utils.losses import FocalLoss, AdaptiveFocalLoss

                custom_objects = {
                    'SpecAugment': SpecAugment,
                    'FocalLoss': FocalLoss,
                    'AdaptiveFocalLoss': AdaptiveFocalLoss,
                    'DropPath': DropPath
                }

                pretrained_model = keras.models.load_model(
                    self.pretrained_model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                self.classifier_model.model.set_weights(pretrained_model.get_weights())
                print("  Successfully loaded pretrained weights")
            except Exception as e:
                print(f"  Warning: Could not load pretrained weights: {e}")
                print("  Starting from random initialization")

        self.classifier_model.compile_model(
            learning_rate=self.config.initial_learning_rate,
            class_weights=class_weights,
            use_focal_loss=True
        )

        self._print_model_architecture()
        
    def _print_model_architecture(self):
        print("\nModel Summary:")
        print(self.classifier_model.get_model_summary())
        
    def execute_training(self, training_features, training_labels,
                         validation_features, validation_labels):
        print(f"\nStarting training for {self.config.training_epochs} epochs...")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Using mixup: {self.config.enable_mixup_augmentation}")

        checkpoint_path = str(self.config.model_output_directory / 'checkpoint_best.keras')

        training_history = self.classifier_model.train(
            train_data=(training_features, training_labels),
            val_data=(validation_features, validation_labels),
            epochs=self.config.training_epochs,
            batch_size=self.config.batch_size,
            use_mixup=self.config.enable_mixup_augmentation,
            checkpoint_path=checkpoint_path
        )

        return training_history
    
    def save_trained_model(self):
        model_save_path = self.config.model_output_directory / 'gtzan_classifier_final.keras'
        self.classifier_model.save_model(model_save_path)
        print(f"\nModel saved to: {model_save_path}")
        
    def get_model(self):
        return self.classifier_model
