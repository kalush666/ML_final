import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class AudioFeatureExtractor:

    def __init__(self,
                 sample_rate: int = 22050,
                 duration: float = 30.0,
                 n_mfcc: int = 20,
                 n_mels: int = 128,
                 n_chroma: int = 12):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_chroma = n_chroma

    def load_audio(self, file_path: Path) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        return audio

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )
        return mfcc.T

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T

    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_chroma=self.n_chroma
        )
        return chroma.T

    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'mfcc': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'chroma': self.extract_chroma(audio)
        }

    def extract_combined_features(self, audio: np.ndarray) -> np.ndarray:
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        chroma = self.extract_chroma(audio)

        min_length = min(mfcc.shape[0], mel_spec.shape[0], chroma.shape[0])

        mfcc = mfcc[:min_length, :]
        mel_spec = mel_spec[:min_length, :]
        chroma = chroma[:min_length, :]

        combined = np.concatenate([mfcc, mel_spec, chroma], axis=1)
        return combined

    def process_file(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        audio = self.load_audio(file_path)
        combined_features = self.extract_combined_features(audio)
        all_features = self.extract_all_features(audio)
        return combined_features, all_features
