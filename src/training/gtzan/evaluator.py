import numpy as np
from pathlib import Path

from training.metrics import MetricsTracker
from .config import GTZANConfig


class GTZANModelEvaluator:
    
    def __init__(self, config: GTZANConfig):
        self.config = config
        self.metrics_tracker = MetricsTracker(
            config.genre_names, 
            config.model_output_directory
        )
        
    def evaluate_model_performance(self, classifier_model, test_features, 
                                    test_labels, training_history):
        print("\nEvaluating on test set...")
        
        test_evaluation_metrics = classifier_model.evaluate((test_features, test_labels))
        print(f"Test metrics: {test_evaluation_metrics}")
        
        prediction_probabilities = classifier_model.predict(
            test_features, 
            use_tta=self.config.enable_test_time_augmentation
        )
        
        predicted_class_indices = np.argmax(prediction_probabilities, axis=1)
        actual_class_indices = np.argmax(test_labels, axis=1)
        
        self._generate_evaluation_artifacts(
            actual_class_indices, 
            predicted_class_indices, 
            training_history
        )
        
        f1_score_metrics = self.metrics_tracker.calculate_f1_scores(
            actual_class_indices, 
            predicted_class_indices
        )
        
        training_summary = self.metrics_tracker.save_training_summary(
            training_history, 
            test_evaluation_metrics, 
            f1_score_metrics
        )
        
        self._print_evaluation_summary(
            training_summary, 
            test_evaluation_metrics, 
            f1_score_metrics
        )
        
        return test_evaluation_metrics, f1_score_metrics
    
    def _generate_evaluation_artifacts(self, true_labels, predicted_labels, 
                                        training_history):
        print("\nGenerating metrics and plots...")
        
        self.metrics_tracker.plot_training_history(training_history)
        self.metrics_tracker.plot_confusion_matrix(true_labels, predicted_labels, normalize=True)
        self.metrics_tracker.plot_confusion_matrix(true_labels, predicted_labels, normalize=False)
        self.metrics_tracker.generate_classification_report(true_labels, predicted_labels)
        
    def _print_evaluation_summary(self, training_summary, test_metrics, f1_metrics):
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Best Val Accuracy: {training_summary['best_val_accuracy']:.4f}")
        print(f"Test Accuracy:     {test_metrics['accuracy']:.4f}")
        print(f"Test Loss:         {test_metrics['loss']:.4f}")
        print(f"F1 Macro:          {f1_metrics['f1_macro']:.4f}")
        print(f"F1 Weighted:       {f1_metrics['f1_weighted']:.4f}")
        print("=" * 50)
