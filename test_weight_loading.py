import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tensorflow import keras
from models.genre_classifier_v2 import GenreCNNClassifierV2
from models.layers.custom_layers import SpecAugment, DropPath
from models.training_utils.losses import FocalLoss, AdaptiveFocalLoss

print("=" * 60)
print("Testing Pretrained Weight Loading")
print("=" * 60)

input_shape = (1292, 164)
num_classes = 10

print("\n1. Building V6 model...")
v6_model = GenreCNNClassifierV2(
    num_classes=num_classes,
    input_shape=input_shape,
    dropout_rate=0.56,
    l2_reg=0.00022,
    use_augmentation=True,
    focal_gamma=2.5,
    label_smoothing=0.09,
    use_adaptive_loss=True,
    per_class_gamma=[2.0, 2.0, 3.0, 4.0, 2.5, 2.0, 2.5, 5.5, 2.5, 8.0],
    confidence_penalty=0.20
)
v6_model.build_model()
print("   V6 model built successfully")

pretrained_path = Path('models/gtzan_classifier_v4/gtzan_classifier_final.keras')

if not pretrained_path.exists():
    print(f"\nERROR: V4 model not found at {pretrained_path}")
    sys.exit(1)

print(f"\n2. Loading V4 pretrained model from: {pretrained_path}")
try:
    custom_objects = {
        'SpecAugment': SpecAugment,
        'FocalLoss': FocalLoss,
        'AdaptiveFocalLoss': AdaptiveFocalLoss,
        'DropPath': DropPath
    }

    v4_model = keras.models.load_model(
        pretrained_path,
        custom_objects=custom_objects,
        compile=False
    )
    print("   V4 model loaded successfully")

    print("\n3. Transferring weights from V4 to V6...")
    v6_model.model.set_weights(v4_model.get_weights())
    print("   Weights transferred successfully")

    print("\n4. Verifying weight transfer...")
    import numpy as np

    test_input = np.random.randn(1, *input_shape).astype(np.float32)

    v4_output = v4_model.predict(test_input, verbose=0)
    v6_output = v6_model.model.predict(test_input, verbose=0)

    max_diff = np.max(np.abs(v4_output - v6_output))
    print(f"   Max prediction difference: {max_diff:.6f}")

    if max_diff < 1e-5:
        print("   SUCCESS - Weight transfer complete, predictions match")
    else:
        print("   FAILED - Weight transfer issue, predictions differ significantly")

    print("\n5. Compiling V6 with new hyperparameters...")
    v6_model.compile_model(
        learning_rate=0.00013,
        class_weights=None,
        use_focal_loss=True
    )
    print("   V6 compiled successfully")

    print("\n" + "=" * 60)
    print("Weight loading test COMPLETED")
    print("=" * 60)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
