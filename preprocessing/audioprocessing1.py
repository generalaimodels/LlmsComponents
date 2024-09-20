
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
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

class AudioCaptionDataset(Dataset):
    """
    PyTorch Dataset for loading and processing audio files with their corresponding captions.

    Args:
        audio_paths (List[str]): List of paths to audio files.
        captions (List[str]): List of corresponding captions.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing captions.
        sample_rate (Optional[int]): Desired sample rate for audio files.
        channels (Optional[int]): Desired number of audio channels.
        duration (Optional[float]): Desired duration of audio clips in seconds.
        normalize (bool): Whether to normalize audio waveforms.
        augmentations (Optional[List[Callable[[Tensor], Tensor]]]): List of augmentation functions.
        feature_extractor (Optional[Callable[[Tensor], Tensor]]): Function to extract features from audio.
    """

    def __init__(
        self,
        audio_paths: List[str],
        captions: List[str],
        tokenizer: PreTrainedTokenizer,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        duration: Optional[float] = None,
        normalize: bool = False,
        augmentations: Optional[List[Callable[[Tensor], Tensor]]] = None,
        feature_extractor: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        if len(audio_paths) != len(captions):
            raise ValueError("The number of audio paths must match the number of captions.")

        self.audio_paths = audio_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.audio_pipeline = AudioPipeline(
            file_paths=audio_paths,
            sample_rate=sample_rate,
            channels=channels,
            duration=duration,
            normalize=normalize,
            augmentations=augmentations,
            feature_extractor=feature_extractor,
            batch_size=1,  # Process one file at a time in __getitem__
            num_workers=1,  # Single worker per instance
        )

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Any]:
        try:
            audio_path = self.audio_paths[index]
            caption = self.captions[index]

            # Process audio file
            audio_tensor = self.audio_pipeline.process_files()[index]
            if audio_tensor is None:
                raise ValueError(f"Failed to process audio file: {audio_path}")

            # Tokenize caption
            encoded_caption = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=128,  # Arbitrary max length; adjust as needed
                return_tensors='pt',
            )
            input_ids = encoded_caption['input_ids'].squeeze(0)
            attention_mask = encoded_caption['attention_mask'].squeeze(0)

            return audio_tensor, input_ids, attention_mask

        except Exception as e:
            warnings.warn(f"Error processing data at index {index}: {e}")
            # Return dummy data or handle as per application needs
            dummy_audio = torch.zeros(1, 16000)  # Adjust shape as needed
            dummy_input_ids = torch.zeros(128, dtype=torch.long)
            dummy_attention_mask = torch.zeros(128, dtype=torch.long)
            return dummy_audio, dummy_input_ids, dummy_attention_mask


def collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Collate function to combine a list of samples into a batch.

    Args:
        batch (List[Tuple[Tensor, Tensor, Tensor]]): List of samples.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Batched audio tensors, input IDs, and attention masks.
    """
    audios, input_ids_list, attention_masks_list = zip(*batch)

    # Pad audio tensors to the same length
    audio_lengths = [audio.size(1) for audio in audios]
    max_audio_length = max(audio_lengths)
    padded_audios = [
        torch.nn.functional.pad(audio, (0, max_audio_length - audio.size(1)))
        if audio.size(1) < max_audio_length else audio
        for audio in audios
    ]
    audio_batch = torch.stack(padded_audios)

    # Stack tokenized captions
    input_ids_batch = torch.stack(input_ids_list)
    attention_masks_batch = torch.stack(attention_masks_list)

    return audio_batch, input_ids_batch, attention_masks_batch


def main():
    """
    Main function to setup data loaders and demonstrate processing.
    """
    try:
        # Example lists of audio file paths and captions
        audio_paths = ['/path/to/audio1.wav', '/path/to/audio2.wav', '/path/to/audio3.wav']
        captions = ['Caption for audio1', 'Caption for audio2', 'Caption for audio3']

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        # Create dataset
        dataset = AudioCaptionDataset(
            audio_paths=audio_paths,
            captions=captions,
            tokenizer=tokenizer,
            sample_rate=16000,
            channels=1,
            duration=5.0,
            normalize=True,
        )

        # Data loader for batching
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Iterate over data loader
        for batch_idx, (audio_batch, input_ids_batch, attention_masks_batch) in enumerate(data_loader):
            # Here you would pass the data to your model
            print(f"Batch {batch_idx}:")
            print(f"Audio batch shape: {audio_batch.shape}")
            print(f"Input IDs batch shape: {input_ids_batch.shape}")
            print(f"Attention masks batch shape: {attention_masks_batch.shape}")

            # Break after one batch for demonstration purposes
            break

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()