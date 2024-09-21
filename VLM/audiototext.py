
import os
import re
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd


class AudioTextDataset(Dataset):
    """
    A Dataset class to handle audio and text data for audio-to-text tasks.

    Attributes:
        dataframe (pd.DataFrame): The dataframe containing audio paths and corresponding text.
        max_length (int): The maximum length for text sequences.
        special_tokens (Dict[str, str]): Special tokens to be added to text sequences.
        padding_value (int): The value used for padding sequences.
        augmentations (Optional[torch.nn.Module]): Augmentations to be applied on audio.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_length: int = 128,
        special_tokens: Optional[Dict[str, str]] = None,
        padding_value: int = 0,
        augmentations: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Initialize the dataset with data and preprocessing configurations.

        Args:
            dataframe (pd.DataFrame): DataFrame with 'audio_paths' and 'text' columns.
            max_length (int, optional): Maximum length of tokenized text sequences. Defaults to 128.
            special_tokens (Optional[Dict[str, str]], optional): Special tokens for text. Defaults to None.
            padding_value (int, optional): Padding value for sequences. Defaults to 0.
            augmentations (Optional[torch.nn.Module], optional): Audio augmentations. Defaults to None.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.special_tokens = special_tokens or {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
        self.padding_value = padding_value
        self.augmentations = augmentations

        # Build vocabulary from the text data
        self.vocab = self.build_vocab(self.dataframe['text'])
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        # Initialize audio feature extractor
        self.feature_extractor = torchaudio.transforms.MelSpectrogram()

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get the sample corresponding to the given index after processing.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: Dictionary containing processed audio features and tokenized text tensors.
        """
        sample = self.dataframe.iloc[idx]

        # Process audio
        audio_path = sample['audio_paths']
        audio_tensor = self.load_audio(audio_path)

        # Apply augmentations if any
        if self.augmentations:
            audio_tensor = self.augmentations(audio_tensor)

        # Extract features
        audio_features = self.extract_features(audio_tensor)

        # Process text
        text = sample['text']
        input_ids, masked_ids = self.process_text(text)

        return {
            'audio_features': audio_features,
            'input_ids': input_ids,
            'masked_ids': masked_ids
        }

    @staticmethod
    def build_vocab(texts: pd.Series) -> List[str]:
        """
        Build a vocabulary from a list of texts.

        Args:
            texts (pd.Series): Series of text data.

        Returns:
            List[str]: List of unique tokens.
        """
        tokens = set()
        for text in texts:
            tokens.update(re.findall(r'\w+', text.lower()))
        return sorted(tokens)

    def load_audio(self, path: str) -> torch.Tensor:
        """
        Load an audio file.

        Args:
            path (str): Path to the audio file.

        Returns:
            torch.Tensor: Loaded audio tensor.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If there is an error loading the audio file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {path}: {e}")
        return waveform

    def extract_features(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the audio tensor.

        Args:
            audio_tensor (torch.Tensor): The audio waveform tensor.

        Returns:
            torch.Tensor: Extracted audio features.
        """
        features = self.feature_extractor(audio_tensor)
        return features.squeeze(0)  # Remove channel dimension if it's single-channel audio

    def process_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize and process text data.

        Args:
            text (str): The input text.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors of input_ids and masked_ids.
        """
        # Add special tokens
        tokens = [self.special_tokens['bos_token']] + re.findall(r'\w+', text.lower()) + [self.special_tokens['eos_token']]

        # Convert tokens to indices
        input_ids = [self.token2idx.get(token, self.token2idx.get(self.special_tokens['pad_token'])) for token in tokens]

        # Truncate or pad sequences
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.token2idx.get(self.special_tokens['pad_token'])] * padding_length

        # Create masked ids (e.g., for language modeling tasks)
        masked_ids = self.create_masked_ids(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(masked_ids, dtype=torch.long)

    def create_masked_ids(self, input_ids: List[int]) -> List[int]:
        """
        Create masked ids for masked language modeling.

        Args:
            input_ids (List[int]): List of input token ids.

        Returns:
            List[int]: List of masked token ids.
        """
        # For simplicity, let's mask tokens at random positions
        masked_ids = input_ids.copy()
        # Implement masking logic if required
        return masked_ids


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle batching of variable-length sequences.

    Args:
        batch (List[Dict[str, Any]]): List of samples.

    Returns:
        Dict[str, torch.Tensor]: Batched tensors.
    """
    audio_features = [item['audio_features'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    masked_ids = [item['masked_ids'] for item in batch]

    # Pad audio features
    audio_features_padded = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True)

    # Stack input_ids and masked_ids
    input_ids_tensor = torch.stack(input_ids)
    masked_ids_tensor = torch.stack(masked_ids)

    return {
        'audio_features': audio_features_padded,
        'input_ids': input_ids_tensor,
        'masked_ids': masked_ids_tensor
    }


def get_dataloader(
    dataframe: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataframe (pd.DataFrame): The dataframe containing data.
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle data. Defaults to True.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        **dataset_kwargs: Additional arguments for the dataset.

    Returns:
        DataLoader: DataLoader object.
    """
    dataset = AudioTextDataset(dataframe, **dataset_kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader
