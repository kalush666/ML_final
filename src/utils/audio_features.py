import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import signal
import sys
import threading
from functools import wraps


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Audio loading timed out")


def timeout_windows(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException("Audio loading timed out")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        
        return wrapper
    return decorator


class AudioFeatureExtractor:

    def __init__(self,
                 sample_rate: int = 22050,
                 duration: float = 30.0,
                 n_mfcc: int = 20,
                 n_mels: int = 128,
                 n_chroma: int = 12,
                 load_timeout: int = 60):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.load_timeout = load_timeout

    def _load_audio_impl(self, file_path: Path, offset: float = 0.0) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=self.sample_rate, 
                                duration=self.duration, offset=offset)
        return audio

    def load_audio(self, file_path: Path, offset: float = 0.0) -> np.ndarray:
        if sys.platform == 'win32':
            result = [TimeoutException("Audio loading timed out")]
            
            def load_thread():
                try:
                    result[0] = self._load_audio_impl(file_path, offset)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=load_thread, daemon=True)
            thread.start()
            thread.join(timeout=self.load_timeout)
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        else:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.load_timeout)
            try:
                audio, _ = librosa.load(file_path, sr=self.sample_rate, 
                                        duration=self.duration, offset=offset)
                return audio
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def _load_audio_full_impl(self, file_path: Path) -> Tuple[np.ndarray, float]:
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        duration = len(audio) / self.sample_rate
        return audio, duration

    def load_audio_full(self, file_path: Path) -> Tuple[np.ndarray, float]:
        if sys.platform == 'win32':
            result = [TimeoutException("Audio loading timed out")]
            
            def load_thread():
                try:
                    result[0] = self._load_audio_full_impl(file_path)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=load_thread, daemon=True)
            thread.start()
            thread.join(timeout=self.load_timeout)
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        else:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.load_timeout)
            try:
                audio, _ = librosa.load(file_path, sr=self.sample_rate)
                duration = len(audio) / self.sample_rate
                return audio, duration
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def extract_multi_segments(self, file_path: Path, n_segments: int = 3,
                                random_offset: bool = True) -> list:
        audio_full, total_duration = self.load_audio_full(file_path)
        
        if total_duration <= self.duration:
            features = self.extract_combined_features(audio_full)
            return [features] * n_segments
        
        segments = []
        if random_offset:
            max_offset = total_duration - self.duration
            offsets = np.random.uniform(0, max_offset, n_segments)
        else:
            offsets = np.linspace(0, total_duration - self.duration, n_segments)
        
        for offset in offsets:
            start_sample = int(offset * self.sample_rate)
            end_sample = start_sample + int(self.duration * self.sample_rate)
            audio_segment = audio_full[start_sample:end_sample]
            
            if len(audio_segment) < int(self.duration * self.sample_rate):
                audio_segment = np.pad(audio_segment, 
                    (0, int(self.duration * self.sample_rate) - len(audio_segment)))
            
            features = self.extract_combined_features(audio_segment)
            segments.append(features)
        
        return segments

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

    def extract_combined_features(self, audio: np.ndarray, target_length: int = 1292) -> np.ndarray:
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        chroma = self.extract_chroma(audio)

        min_length = min(mfcc.shape[0], mel_spec.shape[0], chroma.shape[0])

        mfcc = mfcc[:min_length, :]
        mel_spec = mel_spec[:min_length, :]
        chroma = chroma[:min_length, :]

        combined = np.concatenate([mfcc, mel_spec, chroma], axis=1)

        if combined.shape[0] < target_length:
            pad_width = target_length - combined.shape[0]
            combined = np.pad(combined, ((0, pad_width), (0, 0)), mode='constant')
        elif combined.shape[0] > target_length:
            combined = combined[:target_length, :]

        return combined

    def process_file(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        audio = self.load_audio(file_path)
        combined_features = self.extract_combined_features(audio)
        all_features = self.extract_all_features(audio)
        return combined_features, all_features
