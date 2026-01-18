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
                 include_rhythm: bool = True,
                 include_rock_features: bool = False,
                 load_timeout: int = 60):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.include_rhythm = include_rhythm
        self.include_rock_features = include_rock_features
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
    
    def extract_rock_discriminative_features(self, audio: np.ndarray) -> np.ndarray:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)

        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate, onset_envelope=onset_env)

        hop_length = 512
        n_frames = len(spectral_flux)

        window_size = int(4.0 * self.sample_rate / hop_length)
        if window_size >= n_frames:
            window_size = max(1, n_frames // 4)

        tempo_variance = np.zeros(n_frames)
        for i in range(n_frames):
            start = max(0, i - window_size // 2)
            end = min(n_frames, i + window_size // 2)
            window = onset_env[start:end]
            if len(window) > 1:
                tempo_variance[i] = np.std(window)

        return np.stack([
            spectral_rolloff / (np.max(spectral_rolloff) + 1e-8),
            spectral_flux / (np.max(spectral_flux) + 1e-8),
            tempo_variance / (np.max(tempo_variance) + 1e-8)
        ], axis=1)

    def extract_rhythm_features(self, audio: np.ndarray) -> np.ndarray:
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)

        tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=self.sample_rate, onset_envelope=onset_env
        )

        spectral_flux = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, aggregate=np.median
        )

        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        contrast_mean = np.mean(contrast, axis=0)

        target_len = len(onset_env)

        def resample_feature(feat, target_len):
            if len(feat) == target_len:
                return feat
            indices = np.linspace(0, len(feat) - 1, target_len).astype(int)
            return feat[indices]

        rms_resampled = resample_feature(rms, target_len)
        zcr_resampled = resample_feature(zcr, target_len)
        contrast_resampled = resample_feature(contrast_mean, target_len)

        base_rhythm = np.stack([
            onset_env / (np.max(onset_env) + 1e-8),
            rms_resampled / (np.max(rms_resampled) + 1e-8),
            zcr_resampled / (np.max(zcr_resampled) + 1e-8),
            contrast_resampled / (np.max(np.abs(contrast_resampled)) + 1e-8)
        ], axis=1)

        if self.include_rock_features:
            rock_features = self.extract_rock_discriminative_features(audio)
            if rock_features.shape[0] != target_len:
                rock_features = resample_feature(rock_features.T, target_len).T
                if rock_features.ndim == 1:
                    rock_features = rock_features.reshape(-1, 1)
            return np.hstack([base_rhythm, rock_features])

        return base_rhythm

    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'mfcc': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'chroma': self.extract_chroma(audio),
            'rhythm': self.extract_rhythm_features(audio) if self.include_rhythm else None
        }

    def extract_combined_features(self, audio: np.ndarray, target_length: int = 1292) -> np.ndarray:
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        chroma = self.extract_chroma(audio)

        min_length = min(mfcc.shape[0], mel_spec.shape[0], chroma.shape[0])

        mfcc = mfcc[:min_length, :]
        mel_spec = mel_spec[:min_length, :]
        chroma = chroma[:min_length, :]

        if self.include_rhythm:
            rhythm = self.extract_rhythm_features(audio)
            if rhythm.shape[0] != min_length:
                indices = np.linspace(0, rhythm.shape[0] - 1, min_length).astype(int)
                rhythm = rhythm[indices, :]
            combined = np.concatenate([mfcc, mel_spec, chroma, rhythm], axis=1)
        else:
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
