import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from pathlib import Path
from typing import List, Dict
import json


class MetricsTracker:

    def __init__(self, class_names: List[str], output_dir: Path):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: Dict, save: bool = True) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(history['loss'], label='Train Loss')
        axes[0, 1].plot(history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(history['precision'], label='Train Precision')
        axes[1, 0].plot(history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(history['recall'], label='Train Recall')
        axes[1, 1].plot(history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             normalize: bool = True,
                             save: bool = True) -> None:
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            suffix = '_normalized' if normalize else ''
            plt.savefig(self.output_dir / f'confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_classification_report(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      save: bool = True) -> Dict:
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        if save:
            with open(self.output_dir / 'classification_report.json', 'w') as f:
                json.dump(report_dict, f, indent=2)

        report_str = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )

        if save:
            with open(self.output_dir / 'classification_report.txt', 'w') as f:
                f.write(report_str)

        return report_dict

    def calculate_f1_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_per_class = f1_score(y_true, y_pred, average=None)

        return {
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': {name: float(score) for name, score in zip(self.class_names, f1_per_class)}
        }

    def save_training_summary(self, history: Dict, test_metrics: Dict, f1_scores: Dict) -> None:
        summary = {
            'final_train_accuracy': float(history['accuracy'][-1]),
            'final_val_accuracy': float(history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(history['val_accuracy'])),
            'final_train_loss': float(history['loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'f1_scores': f1_scores,
            'total_epochs': len(history['accuracy'])
        }

        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return summary
