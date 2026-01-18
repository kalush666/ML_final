from pathlib import Path

from .config import GTZANConfig
from .data_loader import GTZANDatasetLoader
from .trainer import GTZANModelTrainer
from .evaluator import GTZANModelEvaluator


class GTZANTrainingOrchestrator:

    def __init__(self, config: GTZANConfig = None, include_rhythm_features: bool = True,
                 include_rock_features: bool = False, pretrained_model_path: Path = None):
        self.config = config or GTZANConfig()
        self.config.model_output_directory.mkdir(parents=True, exist_ok=True)

        self.dataset_loader = GTZANDatasetLoader(
            self.config,
            include_rhythm_features=include_rhythm_features,
            include_rock_features=include_rock_features
        )
        self.model_trainer = GTZANModelTrainer(self.config, pretrained_model_path)
        self.model_evaluator = GTZANModelEvaluator(self.config)
        
    def execute_full_pipeline(self):
        self._print_pipeline_header()
        
        self._prepare_datasets()
        self._build_and_compile_model()
        training_history = self._run_training()
        self._evaluate_and_save(training_history)
        
        self._print_pipeline_footer()
        
    def _print_pipeline_header(self):
        print("=" * 70)
        print(" " * 15 + "GTZAN GENRE CLASSIFIER TRAINING")
        print("=" * 70)
        print(f"\nDataset: GTZAN (10 genres, 1000 tracks)")
        print(f"Expected accuracy: 80-85%")
        
    def _print_pipeline_footer(self):
        print("\n" + "=" * 70)
        print(" " * 20 + "TRAINING COMPLETE!")
        print("=" * 70)
        
    def _prepare_datasets(self):
        self.dataset_loader.load_all_datasets()
        
    def _build_and_compile_model(self):
        input_shape = self.dataset_loader.get_input_shape()
        class_weights = self.dataset_loader.computed_class_weights
        self.model_trainer.initialize_model(input_shape, class_weights)
        
    def _run_training(self):
        training_features, training_labels = self.dataset_loader.get_training_data()
        validation_features, validation_labels = self.dataset_loader.get_validation_data()
        
        training_history = self.model_trainer.execute_training(
            training_features, training_labels,
            validation_features, validation_labels
        )
        
        return training_history
    
    def _evaluate_and_save(self, training_history):
        test_features, test_labels = self.dataset_loader.get_test_data()
        trained_model = self.model_trainer.get_model()
        
        self.model_evaluator.evaluate_model_performance(
            trained_model,
            test_features,
            test_labels,
            training_history
        )
        
        self.model_trainer.save_trained_model()
