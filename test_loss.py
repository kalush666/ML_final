"""Quick test to check loss value - runs in ~30 seconds"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.models.training_utils.losses import FocalLoss

# Create random data simulating 15-class classification
np.random.seed(42)
X = np.random.randn(100, 1292, 160).astype(np.float32)  # 100 samples
y = keras.utils.to_categorical(np.random.randint(0, 15, 100), num_classes=15)

# Simple model
model = keras.Sequential([
    keras.layers.Input(shape=(1292, 160)),
    keras.layers.Reshape((1292, 160, 1)),
    keras.layers.Conv2D(16, 3, strides=4, padding='same', activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(15, activation='softmax')
])

# Test with standard loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n" + "="*50)
print("STANDARD CATEGORICAL CROSSENTROPY")
print("="*50)
history = model.fit(X, y, epochs=1, batch_size=32, verbose=1)
print(f"Loss: {history.history['loss'][0]:.4f}")
print(f"Expected: ~2.7 (which is -log(1/15))")

# Test with focal loss from our implementation
print("\n" + "="*50)
print("FOCAL LOSS (from src/models/training_utils/losses.py)")
print("="*50)
try:
    model2 = keras.Sequential([
        keras.layers.Input(shape=(1292, 160)),
        keras.layers.Reshape((1292, 160, 1)),
        keras.layers.Conv2D(16, 3, strides=4, padding='same', activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(15, activation='softmax')
    ])
    
    model2.compile(optimizer='adam', loss=FocalLoss(gamma=2.0, label_smoothing=0.1), metrics=['accuracy'])
    history2 = model2.fit(X, y, epochs=1, batch_size=32, verbose=1)
    print(f"Loss: {history2.history['loss'][0]:.4f}")
    print(f"Expected: ~2-3 (similar to CE, slightly modified by focal weighting)")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50)
print("VERDICT:")
print("="*50)
print("If loss is ~2-3: GOOD")
print("If loss is ~50: BAD - something is wrong")
