
import os
from typing import List, Tuple, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from transformers import AutoTokenizer, BertTokenizer
import torch.nn.functional as F


class VideoTextDataset(Dataset):
    """
    A PyTorch Dataset for loading video and text data with preprocessing for
    video-to-text tasks.

    Attributes:
        df (pd.DataFrame): DataFrame containing video paths and text descriptions.
        video_paths (List[str]): List of paths to video files.
        texts (List[str]): List of text descriptions corresponding to videos.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text data.
        max_length (int): Maximum sequence length for text tokenization.
        num_frames (int): Number of frames to sample from each video.
        frame_size (Tuple[int, int]): Desired frame size (height, width).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        video_column: str,
        text_column: str,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 128,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize the dataset with paths and parameters.

        Args:
            df (pd.DataFrame): DataFrame containing video paths and text descriptions.
            video_column (str): Name of the column in df that contains video paths.
            text_column (str): Name of the column in df that contains text descriptions.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Maximum sequence length for tokenization.
            num_frames (int): Number of frames to sample from each video.
            frame_size (Tuple[int, int]): Desired frame size (height, width).
        """
        if video_column not in df.columns:
            raise ValueError(f"DataFrame must contain the column '{video_column}'.")
        if text_column not in df.columns:
            raise ValueError(f"DataFrame must contain the column '{text_column}'.")

        self.df = df.reset_index(drop=True)
        self.video_paths: List[str] = self.df[video_column].tolist()
        self.texts: List[str] = self.df[text_column].tolist()
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length: int = max_length
        self.num_frames: int = num_frames
        self.frame_size: Tuple[int, int] = frame_size

        # Check if tokenizer has a mask token
        if not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None:
            raise ValueError(
                "The tokenizer does not have a mask token which is "
                "necessary for masked language modeling."
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple containing:
                video (torch.Tensor): Tensor of shape (D, C, H, W).
                input_ids (torch.Tensor): Token IDs of the text (after masking).
                masked_ids (torch.Tensor): Labels for masked language modeling (-100 for unmasked tokens).
        """
        # Load video
        video_path = self.video_paths[idx]
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

        try:
            video, _, _ = read_video(video_path, pts_unit='sec')
        except Exception as e:
            raise IOError(f"Error reading video '{video_path}': {str(e)}")

        # Video shape: (num_frames_video, H, W, C)
        # Permute to (num_frames_video, C, H, W)
        video = video.permute(0, 3, 1, 2)

        # Sample frames
        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
            video = video[indices]
        else:
            # Loop the video to get required frames
            repeats = (self.num_frames + total_frames - 1) // total_frames
            video = video.repeat(repeats, 1, 1, 1)[:self.num_frames]

        # Apply video transformations
        video = self.apply_transforms(video)

        # Tokenize text
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze(0)  # Shape: (seq_len)

        # Mask tokens for masked language modeling
        input_ids, masked_ids = self.mask_tokens(input_ids)

        return video, input_ids, masked_ids

    def apply_transforms(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply transformations to the video tensor.

        Args:
            video (torch.Tensor): Video tensor of shape (D, C, H, W)

        Returns:
            torch.Tensor: Transformed video tensor.
        """
        # Convert to float and scale to [0, 1]
        video = video.float() / 255.0

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            video = video.flip(-1)  # Flip width dimension

        # Resize video
        video = F.interpolate(
            video, size=self.frame_size, mode='bilinear', align_corners=False
        )

        # Normalize video
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video = (video - mean) / std

        return video

    def mask_tokens(
        self, inputs: torch.Tensor, mlm_probability: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.

        Args:
            inputs (torch.Tensor): Input token IDs.
            mlm_probability (float): Probability of masking a token.

        Returns:
            Tuple containing:
                inputs (torch.Tensor): Masked input IDs.
                labels (torch.Tensor): Labels for masked language modeling.
        """
        labels = inputs.clone()
        # Special tokens should not be masked
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            inputs.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # Create a probability matrix for masking
        probability_matrix = torch.full(labels.shape, mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, replace masked input tokens with random tokens
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest 10% of the time, keep the original tokens

        return inputs, labels


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch (List): List of tuples (video, input_ids, masked_ids).

    Returns:
        Tuple containing batched videos, input_ids, and masked_ids.
    """
    videos, input_ids, masked_ids = zip(*batch)
    videos = torch.stack(videos)
    input_ids = torch.stack(input_ids)
    masked_ids = torch.stack(masked_ids)
    return videos, input_ids, masked_ids