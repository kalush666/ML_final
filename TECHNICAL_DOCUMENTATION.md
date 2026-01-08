# Genre Classifier V2 - Technical Documentation

## Overview

This is an advanced CNN-based music genre classifier that classifies audio into 15 genres using mel-spectrograms. The model achieves ~70% test accuracy with strong generalization through advanced regularization and augmentation techniques.

---

## Architecture Components

### 1. Core Model (`genre_classifier_v2.py`)

**Main Class: `GenreCNNClassifierV2`**

Inherits from `BaseClassifier` and implements a CNN with residual connections, attention mechanisms, and dual pooling.

**Key Parameters:**

- `num_classes`: Number of genre categories (15)
- `input_shape`: Mel-spectrogram dimensions (1292, 160)
- `dropout_rate`: 0.5 - Aggressive dropout for regularization
- `l2_reg`: 0.02 - L2 weight decay coefficient
- `use_augmentation`: True - Enables SpecAugment during training
- `focal_gamma`: 2.0 - Focal loss focusing parameter
- `label_smoothing`: 0.1 - Label smoothing factor

**Architecture Breakdown:**

```
Input (1292, 160)
  ↓
Reshape to (1292, 160, 1)
  ↓
SpecAugment (training only) - Masks time/frequency regions
  ↓
Stem Block:
  - Conv2D(32, 7×7, stride=2) + BN + ReLU + MaxPool(2×2)
  ↓
Residual Tower:
  - ResBlock(64) + MaxPool  → 323×40×64
  - ResBlock(128) + MaxPool → 161×20×128
  - ResBlock(256) + MaxPool → 80×10×256
  - ResBlock(256)           → 80×10×256
  ↓
Head:
  - Global Avg Pooling + Global Max Pooling (concatenated) → 512
  - Dense(512) + BN + ReLU + Dropout(0.5)
  - Dense(256) + BN + ReLU + Dropout(0.5)
  ↓
Output: Dense(15, softmax)
```

### Detailed Layer-by-Layer Explanation

#### Layer 1: Input and Reshape

**Input Shape:** (1292, 160)

- **1292** = Time steps (number of audio frames, ~30 seconds at 22050 Hz)
- **160** = Mel-frequency bins (frequency resolution)

**Reshape Layer:** (1292, 160, 1)

- Adds channel dimension required by Conv2D
- Think of it as grayscale image: Height × Width × Channels
- Here: Time × Frequency × 1 channel

**Purpose:** Mel-spectrogram is a 2D representation where:

- Horizontal axis = time progression
- Vertical axis = frequency (low bass → high treble)
- Pixel intensity = energy at that time-frequency point

---

#### Layer 2: SpecAugment (Training Only)

**Type:** Custom Keras Layer  
**Activation:** Only during training (`training=True`)

**What it does:**

1. **Frequency Masking:** Randomly zeros out 2 horizontal bands (max 20 bins each)

   - Example: Mask frequencies 50-70 and 120-140
   - Simulates missing frequency information (like bass dropout)

2. **Time Masking:** Randomly zeros out 2 vertical bands (max 40 frames each)
   - Example: Mask times 100-140 and 500-540
   - Simulates audio cuts or missing segments

**Why this helps:**

- Forces model to not rely on specific frequencies or time windows
- Prevents overfitting to exact spectral patterns
- Model learns robust features that work even with missing data
- Similar to dropout but spatially structured

**Example:** If model always used "kick drum at 2 seconds" to detect Rock, masking that region forces it to learn alternative features.

---

#### Layer 3: Stem Block

**Purpose:** Initial feature extraction with aggressive downsampling

##### 3a. Conv2D(32 filters, 7×7 kernel, stride=2)

- **Input:** (1292, 160, 1)
- **Output:** (646, 80, 32)
- **Parameters:** 32 × (7×7×1 + 1) = 1,600

**What it does:**

- Large 7×7 receptive field captures both temporal and frequency patterns
- Stride=2 downsamples by 2× (reduces computation and overfitting)
- 32 filters learn 32 different patterns:
  - Filter 1: Might detect "rising pitch"
  - Filter 2: Might detect "harmonic structure"
  - Filter 3: Might detect "rhythmic repetition"
  - Etc.

**Visualization:** Each filter slides across time-frequency plane, producing activation when it matches learned pattern.

##### 3b. Batch Normalization

- **Input:** (646, 80, 32)
- **Output:** (646, 80, 32)
- **Parameters:** 128 (2 per channel: γ, β)

**What it does:**

- Normalizes each channel to mean=0, std=1
- Stabilizes training by preventing "internal covariate shift"
- Allows higher learning rates
- Acts as mild regularization

**Math:** For each channel c: `y = γ * (x - μ) / σ + β`

##### 3c. ReLU Activation

- **Formula:** `f(x) = max(0, x)`
- **Purpose:**
  - Introduces non-linearity (allows learning complex functions)
  - Computationally efficient
  - Sparse activations (many zeros)

##### 3d. MaxPooling2D(2×2)

- **Input:** (646, 80, 32)
- **Output:** (323, 40, 32)

**What it does:**

- Divides input into 2×2 blocks, takes maximum value
- Further downsampling (reduces size by 4×)
- Provides spatial invariance (small shifts don't affect output)
- Reduces parameters in subsequent layers

---

#### Layer 4-7: Residual Tower (4 Residual Blocks)

Each residual block follows the pattern:

```
x → Conv2D → BN → ReLU → SpatialDropout2D → Conv2D → BN → SE Block → Add(x) → ReLU
    └─────────────────────────────────────────────────────────────────┘
                        (Skip Connection)
```

##### Block 1: ResBlock(64 filters) + MaxPool

**Input:** (323, 40, 32) → **Output:** (161, 20, 64)

**Inside the block:**

1. **Conv2D(64, 3×3):** Learn 64 feature maps
   - Detects patterns like "onset + sustained note"
   - Small 3×3 kernel for local patterns
2. **BatchNorm + ReLU:** Normalize and activate

3. **SpatialDropout2D(0.5):**
   - Randomly drops entire feature maps (50% probability)
   - More effective than pixel dropout for CNNs
   - Example: Drop filters 5, 12, 33, keep others
4. **Conv2D(64, 3×3):** Second convolution for refinement

5. **BatchNorm:** Normalize before attention

6. **Squeeze-Excitation Block:**
   - Global Average Pool → Dense(4) → ReLU → Dense(64) → Sigmoid
   - Output: 64 attention weights (one per channel)
   - Example weights: [0.9, 0.3, 0.8, ..., 0.1]
   - Multiply: Channel 0 × 0.9, Channel 1 × 0.3, etc.
   - **Effect:** Amplifies important channels, suppresses noise
7. **Skip Connection (Add):**
   - Adds original input to processed output
   - **Critical:** Allows gradients to flow directly backward
   - Prevents vanishing gradients in deep networks
   - **Math:** `output = ReLU(F(x) + x)`
8. **Final ReLU:** Non-linearity on combined signal

**MaxPooling(2×2):** (161, 20, 64) after pooling

**What this block learns:**

- Low-level temporal patterns (note transitions)
- Basic frequency relationships (harmonics)
- Rhythmic patterns over ~0.5 second windows

##### Block 2: ResBlock(128 filters) + MaxPool

**Input:** (161, 20, 64) → **Output:** (80, 10, 128)

**Differences from Block 1:**

- 128 filters (more capacity)
- Operating on downsampled input (lower resolution, wider context)

**What this block learns:**

- Mid-level patterns like "verse structure" or "chorus patterns"
- Instrument textures (guitar distortion vs clean piano)
- Tempo/rhythm at ~1-2 second scale
- Combinations of low-level features

##### Block 3: ResBlock(256 filters) + MaxPool

**Input:** (80, 10, 256) → **Output:** (40, 5, 256)

**What this block learns:**

- High-level genre-specific patterns
- Overall song structure (repeated sections)
- Style characteristics spanning 2-4 seconds
- Complex feature combinations
- Genre signatures: "Electronic = repetitive + synthesized + 4/4 beat"

##### Block 4: ResBlock(256 filters) [No Pooling]

**Input:** (40, 5, 256) → **Output:** (40, 5, 256)

**Why no pooling?**

- Already at low resolution (5 frequency bands)
- Further pooling would lose too much information
- Focuses on refinement, not downsampling

**What this block learns:**

- Abstract genre representations
- Global song characteristics
- Relationships between different temporal sections
- Final feature refinement before classification

**Feature Map Interpretation at this stage:**

- Each of 256 channels represents complex concept
- Channel 1: "Amount of vocal presence"
- Channel 2: "Electronic vs acoustic score"
- Channel 3: "Rhythmic complexity"
- Etc.

---

#### Layer 8: Classification Head

##### 8a. Dual Global Pooling

**Two parallel operations:**

**Global Average Pooling:**

- Input: (40, 5, 256)
- Output: (256,)
- **Computation:** For each channel, average all 40×5 = 200 values
- **Captures:** Overall presence of features across entire song

**Global Max Pooling:**

- Input: (40, 5, 256)
- Output: (256,)
- **Computation:** For each channel, take maximum of all 200 values
- **Captures:** Peak activation (strongest occurrence of feature)

**Concatenation:** (256,) + (256,) = (512,)

**Why both?**

- Average: "Is this feature present throughout?"
- Max: "Does this feature appear strongly anywhere?"
- Example:
  - Avg=0.3, Max=0.9 → Feature appears briefly but strongly (guitar solo)
  - Avg=0.8, Max=0.9 → Feature consistent throughout (electronic beat)

##### 8b. Dense Layer 1 (512 units)

**Input:** (512,) → **Output:** (512,)

**What it does:**

- Learns non-linear combinations of pooled features
- Example: "If electronic=high AND tempo=fast → likely Techno"
- Fully connected: Every input connects to every output
- **Parameters:** 512 × 512 + 512 = 262,656

**Batch Normalization:** Stabilizes these large weight matrices

**ReLU:** Non-linearity

**Dropout(0.5):** Randomly zero 50% of neurons

- Prevents co-adaptation (neurons becoming too dependent)
- Encourages redundancy
- Major regularization technique

##### 8c. Dense Layer 2 (256 units)

**Input:** (512,) → **Output:** (256,)

**What it does:**

- Further refinement and dimensionality reduction
- Learns higher-order genre decision rules
- Prepares for final classification
- **Parameters:** 512 × 256 + 256 = 131,328

**Same pattern:** BN → ReLU → Dropout(0.5)

##### 8d. Output Layer (15 units)

**Input:** (256,) → **Output:** (15,)

**What it does:**

- One neuron per genre class
- **Parameters:** 256 × 15 + 15 = 3,855
- No activation here (applied next)

**Softmax Activation:**

- **Formula:** `softmax(x_i) = exp(x_i) / Σ exp(x_j)`
- Converts raw scores to probabilities
- All outputs sum to 1.0
- Example output:
  ```
  Rock:        0.65 (65% confident)
  Electronic:  0.20
  Pop:         0.08
  Jazz:        0.04
  Others:      0.03
  ```

**Interpretation:**

- Model predicts Rock with 65% confidence
- But considers Electronic as second possibility (20%)
- These probabilities used by Focal Loss during training

---

### Total Parameter Count

- **Stem:** ~1,600
- **ResBlocks:** ~450,000 (majority of parameters)
- **Dense Layers:** ~398,000
- **Total:** ~850,000 trainable parameters

### Information Flow Summary

```
Raw Audio (30 seconds)
    ↓
Mel-Spectrogram: Time-frequency representation
    ↓
SpecAugment: Learn robust features despite missing data
    ↓
Stem: Extract basic time-frequency patterns (1292×160 → 323×40)
    ↓
ResBlock 1 (64): Local patterns, note transitions
    ↓
ResBlock 2 (128): Mid-level structures, textures
    ↓
ResBlock 3 (256): High-level genre patterns
    ↓
ResBlock 4 (256): Abstract genre concepts
    ↓
Global Pooling: Aggregate features across time/frequency
    ↓
Dense Layers: Learn genre decision boundaries
    ↓
Softmax: Probability distribution over 15 genres
```

**Key Design Principles:**

1. **Hierarchical Feature Learning:** Low → Mid → High level
2. **Residual Connections:** Enable deep networks (gradient flow)
3. **Attention Mechanisms:** Focus on important features (SE blocks)
4. **Regularization Everywhere:** Dropout, L2, augmentation, batch norm
5. **Multi-Scale Information:** Dual pooling captures different aspects

**Private Methods:**

- `_build_stem(x)`: Initial feature extraction with large kernel
- `_build_residual_tower(x)`: Stack of 4 residual blocks with increasing filters
- `_build_head(x)`: Classification head with dual pooling
- `_create_callbacks(...)`: Configures training callbacks

---

### 2. Loss Function (`losses.py`)

**Class: `FocalLoss`**

Custom loss function designed for handling class imbalance.

**Formula:**

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:

- `p_t`: Predicted probability for the true class
- `γ (gamma)`: Focusing parameter (default: 2.0)
  - Reduces loss contribution from easy examples
  - Increases focus on hard-to-classify samples
- `α (alpha)`: Class balancing weight (default: 0.25)

**Why Focal Loss?**

- Standard cross-entropy treats all samples equally
- With imbalanced data (Rock: 4242, Easy Listening: 17), easy examples dominate
- Focal loss down-weights easy examples, forcing model to learn minority classes

**Label Smoothing:**

- Converts hard labels (1, 0, 0, ...) to soft labels (0.93, 0.007, 0.007, ...)
- Prevents overconfidence and improves calibration
- Formula: `y_smooth = y * (1 - ε) + ε / num_classes` where ε = 0.1

---

### 3. Data Augmentation

#### A. SpecAugment (`custom_layers.py`)

**Purpose:** Audio-specific augmentation that masks portions of the spectrogram.

**Parameters:**

- `freq_mask_param`: 20 - Max frequency bins to mask
- `time_mask_param`: 40 - Max time steps to mask
- `num_freq_masks`: 2 - Number of frequency masks
- `num_time_masks`: 2 - Number of time masks

**How it works:**

1. Randomly selects frequency range [f0, f0+f]
2. Sets those bins to zero (silence)
3. Repeats for time dimension
4. Only applied during training (`training=True`)

**Why SpecAugment?**

- Simulates occlusions, missing data, noise
- Forces model to learn robust features
- Prevents overfitting to specific frequency patterns
- Originally developed for speech recognition, adapted for music

#### B. Mixup (`data_augmentation.py`)

**Class: `MixupGenerator`**

Creates synthetic training samples by blending pairs of examples.

**Formula:**

```python
x_mixed = λ * x1 + (1 - λ) * x2
y_mixed = λ * y1 + (1 - λ) * y2
```

Where λ ~ Beta(α, α) with α = 0.2

**Example:**

- Sample 1: Rock song with label [1, 0, 0, ...]
- Sample 2: Electronic song with label [0, 1, 0, ...]
- λ = 0.7
- Mixed: 70% Rock + 30% Electronic
- Label: [0.7, 0.3, 0, ...]

**Benefits:**

- Reduces overfitting by creating infinite training variations
- Smooths decision boundaries
- Improves calibration and generalization

#### C. Test-Time Augmentation (TTA)

**Function: `apply_test_time_augmentation()`**

Averages predictions from multiple augmented versions of the input.

**Augmentations:**

1. Original spectrogram
2. Time-reversed spectrogram
3. Spectrogram + small Gaussian noise (σ=0.01)

**Usage:**

- Only during inference, not training
- Increases accuracy by 1-3% typically
- Tradeoff: 3× slower inference

---

### 4. Model Blocks (`model_blocks.py`)

#### A. Residual Block

Standard ResNet-style block with modifications:

```python
def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters) → BN → ReLU → SpatialDropout2D
    x = Conv2D(filters) → BN
    x = SqueezeExcitation(x)  # Attention
    x = Add([x, shortcut])
    x = ReLU
    return x
```

**Key Features:**

- Skip connections prevent vanishing gradients
- SpatialDropout2D drops entire feature maps (better than pixel dropout for CNNs)
- Batch Normalization for stable training
- L2 regularization on conv weights

#### B. Squeeze-and-Excitation (SE) Block

**Channel attention mechanism:**

```python
def squeeze_excitation_block(x, ratio=16):
    1. Global Average Pooling → [batch, channels]
    2. Dense(channels // 16) + ReLU  # Bottleneck
    3. Dense(channels) + Sigmoid     # Attention weights
    4. Multiply with original input  # Reweight channels
```

**What it does:**

- Learns which channels (features) are most important
- Suppresses uninformative channels
- Amplifies useful channels
- Adds minimal parameters (<1% of model)

**Why it helps:**

- Not all learned features are equally useful
- Low-frequency features might be more important than high-frequency
- SE block learns these relationships adaptively

---

### 5. Training Pipeline (`train_genre_v2.py`)

**Class: `ImprovedGenreTrainingPipeline`**

Manages the complete training workflow with advanced techniques.

#### Key Methods:

**1. `oversample_minority_classes(X, y)`**

- Problem: Rock (4242 samples) vs Easy Listening (17 samples)
- Solution: Randomly duplicate minority class samples
- Target: Median class count (~240 samples/class)
- Also undersamples majority if count > 2× median

**2. `filter_top_classes(X, y, n_classes)`**

- Optional: Keep only top N most frequent genres
- Useful when rare classes have too few samples
- Re-encodes labels to 0, 1, 2, ..., N-1

**3. `prepare_datasets()`**

- Loads train/val/test splits
- Applies oversampling to training data only
- Calculates class weights for loss function
- Computes class distribution statistics

**4. `build_and_compile_model()`**

- Initializes GenreCNNClassifierV2
- Chooses between Focal Loss or Categorical Crossentropy
- Sets up AdamW optimizer with:
  - Weight decay: 0.01
  - Gradient clipping: 1.0
  - Initial LR: 0.0005

**5. `train_model()`**

- Creates Mixup dataset if enabled
- Uses callbacks:
  - EarlyStopping (patience: 10)
  - LR scheduler (cosine decay)
  - ModelCheckpoint (saves best model by F1)
  - LearningRateFormatter (pretty printing)

---

### 6. Callbacks (`custom_callbacks.py`)

**Class: `LearningRateFormatter`**

Simple callback to print learning rate in decimal format (not scientific notation).

```
Epoch 1 - Learning Rate: 0.000500
Epoch 2 - Learning Rate: 0.000497
...
```

---

## Training Configuration

### Hyperparameters

```python
epochs = 60
batch_size = 32
learning_rate = 0.0005  # Lower than default for stability
use_oversampling = True
use_mixup = True
use_focal_loss = True
```

### Learning Rate Schedule

**Cosine Decay:**

- Starts at 0.0005
- Decays smoothly to 0.00005 (α=0.1)
- Formula: `lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * step / total_steps))`

**Why Cosine?**

- Smooth decay avoids sudden drops
- Allows fine-tuning in later epochs
- Better than step decay for deep networks

### Optimizer: AdamW

**Adam with Weight Decay**

- Standard Adam updates: `m_t, v_t` (momentum terms)
- Adds explicit L2 regularization on weights
- Decouples weight decay from gradient updates
- More effective than Adam + L2 loss term

**Gradient Clipping:**

- Clips gradients to max norm of 1.0
- Prevents exploding gradients
- Stabilizes training with deep networks

---

## Data Flow

### Training Phase:

```
Audio Files (.mp3)
    ↓
AudioFeatureExtractor.load_audio()  → Raw waveform
    ↓
extract_combined_features()
    ↓
Mel-Spectrogram (1292×160)
    ↓
Oversampling (balance classes)
    ↓
MixupGenerator (on-the-fly augmentation)
    ↓
Model Training Loop:
    - SpecAugment (masking)
    - Forward pass
    - Focal Loss computation
    - Backpropagation
    - AdamW update
    ↓
Callbacks (early stopping, LR schedule, checkpoint)
    ↓
Best Model Saved
```

### Inference Phase:

```
New Audio File
    ↓
Feature Extraction → Mel-Spectrogram
    ↓
(Optional) Test-Time Augmentation:
    - Original
    - Time-reversed
    - + Noise
    ↓
Model Prediction
    ↓
Average predictions (if TTA)
    ↓
Softmax probabilities → Predicted genre
```

---

## Metrics Explained

### During Training:

**Accuracy:**

- % of correct predictions
- Can be misleading with imbalanced data
- High accuracy possible by predicting majority class

**Precision:**

- Of all predictions for class C, how many were correct?
- `TP / (TP + FP)`
- High precision = few false alarms

**Recall:**

- Of all actual instances of class C, how many did we find?
- `TP / (TP + FN)`
- High recall = few misses

**F1 Score (Macro):**

- Harmonic mean of precision and recall
- Computed per-class then averaged
- Treats all classes equally (good for imbalanced data)
- `2 * (precision * recall) / (precision + recall)`

**F1 Score (Weighted):**

- Same as macro but weighted by class frequency
- Reflects performance on common classes more

**AUC (Area Under ROC Curve):**

- Measures ranking quality
- 1.0 = perfect separation
- 0.5 = random guessing
- Independent of classification threshold

---

## Performance Considerations

### Why Model Shows Overfitting?

**Symptoms:**

- Training accuracy: 93%
- Test accuracy: 70%
- Gap of 23% indicates memorization

**Causes:**

1. Limited data (~2000 training samples for 15 classes)
2. Complex model (656K parameters)
3. Some genres very similar (Rock vs Alternative)

**Mitigations Applied:**

- Heavy dropout (0.5)
- L2 regularization (0.02)
- SpecAugment + Mixup
- Early stopping
- Class balancing

### Addressing Class Imbalance

**Problem:**

- Rock: 4242 samples
- Easy Listening: 17 samples
- Ratio: 249:1

**Solutions:**

1. **Oversampling:** Duplicate minority classes
2. **Class Weights:** Weight loss by inverse frequency
3. **Focal Loss:** Focus on hard examples
4. **Top-N Classes:** Consider dropping rare classes

**Result:**

- F1 Macro: 0.42 (treats all classes equally)
- F1 Weighted: 0.72 (favors common classes)
- Gap shows model better at popular genres

---

## File Structure

```
src/models/
├── __init__.py                  # Module exports
├── base_classifier.py           # Abstract base class
├── genre_classifier_v2.py       # Main CNN model
├── losses.py                    # Focal loss implementation
├── custom_layers.py             # SpecAugment layer
├── model_blocks.py              # Residual + SE blocks
├── custom_callbacks.py          # LR formatter callback
└── data_augmentation.py         # Mixup + TTA

src/training/
├── train_genre_v2.py           # Training pipeline
├── data_generator.py           # Feature extraction
└── metrics.py                  # Evaluation metrics

src/utils/
└── audio_features.py           # Mel-spectrogram extraction
```

---

## Best Practices Followed

### SOLID Principles:

1. **Single Responsibility:** Each file has one clear purpose
2. **Open/Closed:** Easy to extend (add new augmentations) without modifying core
3. **Liskov Substitution:** GenreCNNClassifierV2 substitutable for BaseClassifier
4. **Interface Segregation:** Focused interfaces, no fat interfaces
5. **Dependency Inversion:** Depends on abstractions (BaseClassifier)

### ML Best Practices:

1. **Reproducibility:** Random seeds, deterministic operations
2. **Separation of Concerns:** Data loading ≠ Model ≠ Training
3. **Validation:** Separate train/val/test splits
4. **Regularization:** Multiple techniques (dropout, L2, augmentation)
5. **Monitoring:** Callbacks track metrics, save best model
6. **Versioning:** V2 doesn't break V1

---

## Common Issues & Solutions

### Issue 1: Loss starts at ~10

**Cause:** Focal loss miscalculation (summing all classes)
**Fix:** Corrected to use only true class probability

### Issue 2: High accuracy, low recall

**Cause:** Model too conservative due to class imbalance
**Fix:** Focal loss + oversampling + adjust decision threshold

### Issue 3: Overfitting (train>>test)

**Mitigation:** Increased regularization (dropout 0.5, L2 0.02, SpecAugment)

### Issue 4: Terminal display issues

**Not a bug:** VS Code terminal doesn't handle `\r` carriage returns well

---

## Future Improvements

1. **More Data:** Collect more samples for minority classes
2. **Reduce Classes:** Focus on top 8-10 distinguishable genres
3. **Ensemble:** Train multiple models, average predictions
4. **Transfer Learning:** Use pretrained audio models (YAMNet, VGGish)
5. **Attention Mechanisms:** Add temporal attention across time steps
6. **Knowledge Distillation:** Distill from larger model
7. **Hard Negative Mining:** Focus on confusing examples
8. **Multi-Scale Features:** Process at different time scales

---

## References

- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR" (2019)
- Mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
- ResNet: He et al., "Deep Residual Learning for Image Recognition" (2015)
- SE-Net: Hu et al., "Squeeze-and-Excitation Networks" (2018)
- AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)

---

## Quick Start

```bash
# Train model
python src/training/train_genre_v2.py

# Results will be saved to:
# - models/genre_classifier_v2/genre_classifier_v2_final.keras
# - models/checkpoints/genre_best_v2.keras (best by F1)
# - models/genre_classifier_v2/*.png (plots)
# - models/genre_classifier_v2/*.json (metrics)
```

---

**Author:** Improved Genre Classifier V2  
**Date:** January 2026  
**License:** Project-specific
