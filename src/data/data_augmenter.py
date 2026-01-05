import librosa
import numpy as np
from pathlib import Path
from typing import Tuple


class AudioAugmenter:

    @staticmethod
    def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented.astype(audio.dtype)

    @staticmethod
    def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
        return librosa.effects.time_stretch(audio, rate=rate)

    @staticmethod
    def pitch_shift(audio: np.ndarray,
                   sr: int,
                   n_steps: int = 2) -> np.ndarray:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    @staticmethod
    def time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
        return np.roll(audio, shift)

    def augment_audio(self,
                     audio: np.ndarray,
                     sr: int,
                     techniques: list = None) -> list:
        if techniques is None:
            techniques = ['noise', 'time_stretch', 'pitch_shift']

        augmented_samples = []

        if 'noise' in techniques:
            augmented_samples.append(self.add_noise(audio))

        if 'time_stretch' in techniques:
            augmented_samples.append(self.time_stretch(audio, rate=0.9))
            augmented_samples.append(self.time_stretch(audio, rate=1.1))

        if 'pitch_shift' in techniques:
            augmented_samples.append(self.pitch_shift(audio, sr, n_steps=-2))
            augmented_samples.append(self.pitch_shift(audio, sr, n_steps=2))

        if 'time_shift' in techniques:
            augmented_samples.append(self.time_shift(audio))

        return augmented_samples


class DatasetAugmenter:

    def __init__(self, augmenter: AudioAugmenter):
        self.augmenter = augmenter

    def augment_underrepresented_classes(self,
                                        audio_paths: list,
                                        target_count: int,
                                        sr: int = 22050) -> list:
        augmented_paths = []
        current_count = len(audio_paths)

        if current_count >= target_count:
            return []

        samples_needed = target_count - current_count

        for i in range(samples_needed):
            original_path = audio_paths[i % len(audio_paths)]
            audio, _ = librosa.load(original_path, sr=sr)

            augmented_list = self.augmenter.augment_audio(audio, sr)
            augmented_audio = augmented_list[i % len(augmented_list)]

            output_path = Path(original_path).parent / f"{Path(original_path).stem}_aug_{i}.npy"
            np.save(output_path, augmented_audio)
            augmented_paths.append(output_path)

        return augmented_paths
