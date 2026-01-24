import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.audio_features import AudioFeatureExtractor


class AudioService:

    def __init__(self):
        self.genre_extractor = AudioFeatureExtractor(
            sample_rate=22050,
            duration=30.0,
            include_rhythm=True,
            include_rock_features=False
        )
        self.author_extractor = AudioFeatureExtractor(
            sample_rate=22050,
            duration=30.0
        )

    async def save_upload_file(self, upload_file: UploadFile) -> Path:
        suffix = Path(upload_file.filename).suffix or '.mp3'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload_file.read()
            tmp.write(content)
            return Path(tmp.name)

    def extract_genre_features(self, audio_path: Path) -> np.ndarray:
        audio = self.genre_extractor.load_audio(audio_path)
        features = self.genre_extractor.extract_combined_features(audio)
        return features

    def extract_author_features(self, audio_path: Path) -> np.ndarray:
        audio = self.author_extractor.load_audio(audio_path)
        features = self.author_extractor.extract_mel_spectrogram(audio)
        
        target_length = 258
        if features.shape[0] < target_length:
            pad_width = target_length - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        elif features.shape[0] > target_length:
            features = features[:target_length, :]
        
        return features

    def cleanup(self, file_path: Path):
        try:
            file_path.unlink()
        except Exception:
            pass


audio_service = AudioService()
