
import os
import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Optional
from functools import partial
import librosa
import torch
import torchaudio
from torch import Tensor


def process_file_wrapper(
    file_path: str,
    sample_rate: Optional[int],
    channels: Optional[int],
    duration: Optional[float],
    normalize: bool,
    augmentations: List[Callable[[Tensor], Tensor]],
    feature_extractor: Optional[Callable[[Tensor], Tensor]],
) -> Optional[Tensor]:
    """
    Process a single audio file: load, preprocess, augment, and extract features.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (Optional[int]): Desired sample rate.
        channels (Optional[int]): Desired number of channels.
        duration (Optional[float]): Desired duration in seconds.
        normalize (bool): Whether to normalize the audio.
        augmentations (List[Callable[[Tensor], Tensor]]): List of augmentation functions.
        feature_extractor (Optional[Callable[[Tensor], Tensor]]): Feature extraction function.

    Returns:
        Optional[Tensor]: Processed audio tensor or None if processing fails.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load audio file
        waveform, sr = torchaudio.load(file_path)

        # Resample if needed
        if sample_rate and sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate

        # Adjust channels
        if channels:
            if waveform.size(0) != channels:
                if channels == 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                elif channels == 2:
                    if waveform.size(0) == 1:
                        waveform = waveform.repeat(2, 1)
                    else:
                        # Adjust to first two channels
                        waveform = waveform[:2, :]
                else:
                    raise ValueError(f"Unsupported number of channels: {channels}")

        # Trim or pad to duration
        if duration:
            num_samples = int(duration * sr)
            if waveform.size(1) > num_samples:
                waveform = waveform[:, :num_samples]
            elif waveform.size(1) < num_samples:
                padding = num_samples - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Normalize
        if normalize:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

        # Apply augmentations
        for aug in augmentations:
            waveform = aug(waveform)

        # Feature extraction
        if feature_extractor:
            features = feature_extractor(waveform)
        else:
            features = waveform

        return features

    except Exception as e:
        warnings.warn(f"Error processing {file_path}: {e}")
        return None


class AudioPipeline:
    """
    Audio processing pipeline that loads, preprocesses, augments, and extracts features from audio files.
    Utilizes multiprocessing for efficient parallel processing.

    Args:
        file_paths (List[str]): List of audio file paths.
        sample_rate (Optional[int]): Desired sample rate for audio files.
        channels (Optional[int]): Desired number of audio channels.
        duration (Optional[float]): Desired duration of audio clips in seconds.
        normalize (bool): Whether to normalize audio waveforms.
        augmentations (Optional[List[Callable[[Tensor], Tensor]]]): List of augmentation functions.
        feature_extractor (Optional[Callable[[Tensor], Tensor]]): Function to extract features from audio.
        batch_size (int): Number of samples per batch.
        num_workers (Optional[int]): Number of worker processes for multiprocessing.
    """

    def __init__(
        self,
        file_paths: List[str],
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        duration: Optional[float] = None,
        normalize: bool = False,
        augmentations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        feature_extractor: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ):
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration
        self.normalize = normalize
        self.augmentations = augmentations or []
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.num_workers = num_workers or cpu_count()

    def process_files(self) -> List[Tensor]:
        """
        Process all audio files using multiprocessing.

        Returns:
            List[Tensor]: List of processed audio tensors.
        """
        func = partial(
            process_file_wrapper,
            sample_rate=self.sample_rate,
            channels=self.channels,
            duration=self.duration,
            normalize=self.normalize,
            augmentations=self.augmentations,
            feature_extractor=self.feature_extractor,
        )
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(func, self.file_paths)
        processed = [res for res in results if res is not None]
        return processed

    def get_batches(self) -> List[Tensor]:
        """
        Get the processed audio data in batches.

        Returns:
            List[Tensor]: List of batch tensors.
        """
        processed = self.process_files()
        batches = []
        for i in range(0, len(processed), self.batch_size):
            batch = processed[i : i + self.batch_size]
            batch_tensor = torch.stack(batch)
            batches.append(batch_tensor)
        return batches