import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from tensorflow import keras
from models.layers.custom_layers import SpecAugment, DropPath
from models.training_utils.losses import FocalLoss, AdaptiveFocalLoss
from training.gtzan import GTZANConfig, GTZANDatasetLoader

print("=" * 70)
print("Rock Failure Analysis - V4 Model")
print("=" * 70)

v4_model_path = Path('models/gtzan_classifier_v4/gtzan_classifier_final.keras')

if not v4_model_path.exists():
    print(f"ERROR: V4 model not found at {v4_model_path}")
    sys.exit(1)

print("\n1. Loading V4 model...")
custom_objects = {
    'SpecAugment': SpecAugment,
    'FocalLoss': FocalLoss,
    'AdaptiveFocalLoss': AdaptiveFocalLoss,
    'DropPath': DropPath
}

model = keras.models.load_model(v4_model_path, custom_objects=custom_objects)
print("   V4 model loaded successfully")

print("\n2. Loading test data...")
config = GTZANConfig(
    train_csv_path=Path('data/processed/gtzan_splits/train_fixed.csv'),
    validation_csv_path=Path('data/processed/gtzan_splits/val_fixed.csv'),
    test_csv_path=Path('data/processed/gtzan_splits/test_fixed.csv'),
    model_output_directory=Path('models/analysis_temp')
)

loader = GTZANDatasetLoader(config, include_rhythm_features=True)
print("   Loading test dataset...")
loader.load_all_datasets()
test_features, test_labels = loader.get_test_data()

print(f"   Test features shape: {test_features.shape}")
print(f"   Test samples: {len(test_labels)}")

genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']
rock_idx = 9

rock_mask = test_labels == rock_idx
rock_features = test_features[rock_mask]
rock_labels = test_labels[rock_mask]

print(f"   Rock test samples: {len(rock_features)}")

print("\n3. Running V4 predictions on rock samples...")
predictions = model.predict(rock_features, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

correct = predicted_classes == rock_labels
accuracy = np.mean(correct)
num_correct = np.sum(correct)
num_total = len(rock_labels)

print(f"\n{'=' * 70}")
print(f"ROCK CLASSIFICATION RESULTS")
print(f"{'=' * 70}")
print(f"Correct: {num_correct}/{num_total} ({accuracy*100:.1f}%)")
print(f"Failed:  {num_total - num_correct}/{num_total} ({(1-accuracy)*100:.1f}%)")

print(f"\n{'=' * 70}")
print(f"DETAILED FAILURE ANALYSIS")
print(f"{'=' * 70}")

failures = []
for idx, (pred_class, true_class, pred_probs, is_correct) in enumerate(
    zip(predicted_classes, rock_labels, predictions, correct)):

    if not is_correct:
        failures.append({
            'idx': idx,
            'predicted': genre_names[pred_class],
            'confidence': pred_probs[pred_class] * 100,
            'rock_confidence': pred_probs[rock_idx] * 100,
            'probs': pred_probs
        })

print(f"\nFailed Rock Samples ({len(failures)} total):")
print(f"{'-' * 70}")

confusion_counts = {}
for fail in failures:
    pred_genre = fail['predicted']
    confusion_counts[pred_genre] = confusion_counts.get(pred_genre, 0) + 1

    print(f"\nRock Sample #{fail['idx']}")
    print(f"  Predicted: {fail['predicted']} (confidence: {fail['confidence']:.1f}%)")
    print(f"  Rock confidence: {fail['rock_confidence']:.1f}%")

    top_3_indices = np.argsort(fail['probs'])[-3:][::-1]
    print(f"  Top 3 predictions:")
    for i, top_idx in enumerate(top_3_indices, 1):
        print(f"    {i}. {genre_names[top_idx]}: {fail['probs'][top_idx]*100:.1f}%")

print(f"\n{'=' * 70}")
print(f"CONFUSION SUMMARY")
print(f"{'=' * 70}")
print(f"Rock misclassified as:")
for genre, count in sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True):
    pct = (count / len(failures)) * 100 if failures else 0
    print(f"  {genre:12s}: {count:2d} samples ({pct:5.1f}% of failures)")

print(f"\n{'=' * 70}")
print(f"ALL ROCK SAMPLES ANALYSIS")
print(f"{'=' * 70}")

print(f"\nAverage confidence scores across all {num_total} rock samples:")
avg_probs = np.mean(predictions, axis=0)
for genre_idx, genre_name in enumerate(genre_names):
    print(f"  {genre_name:12s}: {avg_probs[genre_idx]*100:.1f}%")

print(f"\nRock confidence distribution:")
rock_confidences = predictions[:, rock_idx]
print(f"  Min:    {np.min(rock_confidences)*100:.1f}%")
print(f"  Max:    {np.max(rock_confidences)*100:.1f}%")
print(f"  Mean:   {np.mean(rock_confidences)*100:.1f}%")
print(f"  Median: {np.median(rock_confidences)*100:.1f}%")

print(f"\n{'=' * 70}")
print(f"INTER-GENRE CONFUSION ANALYSIS")
print(f"{'=' * 70}")

print("\nRunning predictions on ALL test samples...")
all_predictions = model.predict(test_features, verbose=0)
all_predicted = np.argmax(all_predictions, axis=1)

confusion_matrix = np.zeros((10, 10), dtype=int)
for true_label, pred_label in zip(test_labels, all_predicted):
    confusion_matrix[true_label, pred_label] += 1

print("\nGenres most confused with rock:")
rock_row = confusion_matrix[rock_idx, :]
for genre_idx in np.argsort(rock_row)[::-1]:
    if genre_idx != rock_idx and rock_row[genre_idx] > 0:
        print(f"  {genre_names[genre_idx]:12s}: {rock_row[genre_idx]:2d} times")

print("\nGenres misclassified AS rock:")
rock_col = confusion_matrix[:, rock_idx]
for genre_idx in np.argsort(rock_col)[::-1]:
    if genre_idx != rock_idx and rock_col[genre_idx] > 0:
        print(f"  {genre_names[genre_idx]:12s}: {rock_col[genre_idx]:2d} times")

print(f"\n{'=' * 70}")
print(f"Analysis complete!")
print(f"{'=' * 70}")
