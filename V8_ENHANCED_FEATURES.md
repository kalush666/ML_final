# V8 - Enhanced Rock-Discriminative Features

## Problem Analysis

After 7 versions of hyperparameter tuning, rock classification performance consistently degraded:
- V4: Rock F1 0.50 (8/15 correct) - 75.8% overall
- V6: Rock F1 0.44 (6/15 correct) - 77.9% overall
- V7: Rock F1 0.37 (5/15 correct) - 74.5% overall

**Root Cause:** Increasing per-class gamma weights made the model MORE confused about rock, not less. The issue wasn't the loss weighting - it was missing acoustic features that distinguish rock from metal/country/pop.

## Solution: Feature Engineering

### New Rock-Discriminative Features Added

V8 adds 3 new features to capture rock's unique acoustic characteristics:

1. **Spectral Rolloff**
   - Frequency below which 85% of spectral energy is contained
   - Rock has higher energy in high frequencies due to guitar distortion
   - Distinguishes rock's "bright" distorted sound from metal's "heavy" sound

2. **Spectral Flux**
   - Measures how quickly the frequency spectrum changes over time
   - Rock has more dynamic spectrum changes (varied guitar techniques)
   - More variable than metal's consistent heaviness

3. **Tempo Variance**
   - Standard deviation of onset strength in sliding windows
   - Rock has more tempo/energy variability within songs
   - Metal tends to maintain consistent tempo/energy

### Implementation

**Modified File:** `src/utils/audio_features.py`

New method `extract_rock_discriminative_features()` computes these 3 features and integrates them into the rhythm feature extraction pipeline.

**Total Feature Dimensions:** 167 (was 164)
- 20 MFCC
- 128 Mel Spectrogram
- 12 Chroma
- 4 Original Rhythm Features
- **3 New Rock Features** ← NEW

## V8 Configuration

V8 uses V4's proven hyperparameters (V4 remains best at 75.8%) with the enhanced feature set:

```python
model_output_directory: 'models/gtzan_classifier_v8'
initial_learning_rate: 0.00012
dropout_rate: 0.55
l2_regularization: 0.00025
label_smoothing_factor: 0.1
mixup_alpha: 0.4
confidence_penalty: 0.15
per_class_gamma: [2.0, 2.0, 2.5, 3.5, 2.5, 2.0, 2.5, 3.5, 2.5, 4.0]
```

**Key Difference from V4:** Rock gamma = 4.0 (moderate focus) vs V4's ~2.5
- Not extreme like V6's 8.0 or V7's 6.0
- Balanced approach with new features doing the heavy lifting

## Success Criteria

**Minimum Goals:**
- Rock F1 > 0.60 (at least 9/15 correct)
- Overall accuracy > 77%
- No regression in other genres

**Stretch Goals:**
- Rock F1 > 0.65 (10/15 correct)
- Overall accuracy > 78%

## Training from Scratch

V8 trains from **random initialization** (not transfer learning from V4) because:
1. New features change input dimensions (164 → 167)
2. Model needs to learn how to use new rock-discriminative features
3. Transfer learning from V4 would bias toward V4's confusion patterns

## Fallback Plan

If V8 doesn't achieve Rock F1 > 0.60:
- Accept V4 (75.8% accuracy, Rock F1 0.50) as final model
- Document that rock/metal/country discrimination requires additional features beyond standard audio processing
- Recommend genre metadata or user feedback for rock classification in production
