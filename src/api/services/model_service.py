import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tensorflow as tf
from tensorflow import keras

from models.rnn_author_classifier import RNNAuthorClassifier, AttentionPooling, ResidualBiLSTMBlock


GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


class ModelService:

    def __init__(self):
        self.genre_model = None
        self.author_model = None
        self.artist_mapping = None
        self.models_dir = Path(__file__).resolve().parent.parent.parent.parent / 'models'

    def load_genre_model(self) -> bool:
        try:
            model_path = self.models_dir / 'gtzan_classifier_v4' / 'gtzan_classifier_final.keras'
            if not model_path.exists():
                print(f"Genre model not found at {model_path}")
                return False
            self.genre_model = keras.models.load_model(model_path, compile=False)
            print(f"Genre model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading genre model: {e}")
            return False

    def load_author_model(self) -> bool:
        try:
            model_path = self.models_dir / 'author_classifier_rnn' / 'author_classifier_final.keras'
            mapping_path = self.models_dir / 'author_classifier_rnn' / 'artist_mapping.npy'
            
            if not model_path.exists():
                print(f"Author model not found at {model_path}")
                return False

            num_classes = 50
            input_shape = (258, 128)
            
            classifier = RNNAuthorClassifier(
                num_classes=num_classes,
                input_shape=input_shape,
                lstm_units=[128, 64],
                dropout_rate=0.3,
                recurrent_dropout=0.1,
                l2_reg=0.0001,
                use_attention=True,
                use_multihead_attention=True,
                num_attention_heads=4,
                label_smoothing=0.1
            )
            classifier.build_model()
            classifier.compile_model(learning_rate=0.0005)
            
            dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
            _ = classifier.model(dummy_input, training=False)
            
            classifier.model.load_weights(str(model_path))
            self.author_model = classifier.model
            
            if mapping_path.exists():
                self.artist_mapping = np.load(mapping_path, allow_pickle=True).item()
            
            print(f"Author model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading author model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_all_models(self) -> dict:
        return {
            'genre': self.load_genre_model(),
            'author': self.load_author_model()
        }

    def predict_genre(self, features: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.genre_model is None:
            raise RuntimeError("Genre model not loaded")
        
        features = np.expand_dims(features, axis=0)
        predictions = self.genre_model.predict(features, verbose=0)[0]
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        results = [(GENRE_LABELS[i], float(predictions[i])) for i in top_indices]
        
        return results

    def predict_author(self, features: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.author_model is None:
            raise RuntimeError("Author model not loaded")
        
        features = np.expand_dims(features, axis=0)
        predictions = self.author_model.predict(features, verbose=0)[0]
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        if self.artist_mapping and 'idx_to_artist' in self.artist_mapping:
            idx_to_artist = self.artist_mapping['idx_to_artist']
            results = [(idx_to_artist.get(i, f"Artist_{i}"), float(predictions[i])) for i in top_indices]
        else:
            results = [(f"Artist_{i}", float(predictions[i])) for i in top_indices]
        
        return results

    def is_genre_loaded(self) -> bool:
        return self.genre_model is not None

    def is_author_loaded(self) -> bool:
        return self.author_model is not None


model_service = ModelService()
