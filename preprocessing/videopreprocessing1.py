import logging
import os
import tempfile
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union

import requests
import torch
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset
from torchvision.io import read_video
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralizedVideoCaptionDataset(Dataset):
    """
    A generalized dataset loader for video-caption data.
    Supports any number of splits, columns, and data formats.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        video_column: str,
        caption_column: str,
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = True,
        special_tokens: Optional[Dict[str, str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *args,
        **kwargs
    ):
        """
        Initialize the dataset.

        Args:
            data (List[Dict[str, Any]]): List of data entries.
            video_column (str): Key for video paths in data entries.
            caption_column (str): Key for captions in data entries.
            tokenizer (PreTrainedTokenizer): Tokenizer for captions.
            max_length (Optional[int]): Max tokenization length.
            padding (Union[bool, str]): Padding strategy.
            truncation (bool): If True, truncate sequences.
            special_tokens (Optional[Dict[str, str]]): Special tokens.
            transform (Optional[Callable]): Transform function for videos.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.data = data
        self.video_column = video_column
        self.caption_column = caption_column
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.special_tokens = special_tokens or {}
        self.transform = transform

        # Add special tokens to the tokenizer
        if self.special_tokens:
            special_tokens_dict = {'additional_special_tokens': list(self.special_tokens.values())}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Added {num_added_toks} special tokens.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data[idx]
        video_path = entry.get(self.video_column)
        captions = entry.get(self.caption_column)

        if not video_path:
            raise ValueError(f"No video path found for index {idx} in column '{self.video_column}'.")
        if not captions:
            raise ValueError(f"No caption found for index {idx} in column '{self.caption_column}'.")

        video = self._load_video(video_path)
        if self.transform:
            video = self.transform(video)

        tokenized_captions = self._tokenize_captions(captions)

        additional_columns = {k: v for k, v in entry.items() if k not in [self.video_column, self.caption_column]}

        return {'video': video, 'captions': tokenized_captions, **additional_columns}

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load a video from a path or URL.

        Args:
            video_path (str): Local path or URL to the video.

        Returns:
            torch.Tensor: Video tensor.

        Raises:
            ValueError: If the video cannot be loaded.
        """
        try:
            if video_path.startswith(('http://', 'https://')):
                logger.info(f"Downloading video from URL: {video_path}")
                response = requests.get(video_path, stream=True)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_path)[1]) as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    temp_video_path = tmp_file.name
                video, _, _ = read_video(temp_video_path)
                os.remove(temp_video_path)
            else:
                video, _, _ = read_video(video_path)
            return video
        except Exception as e:
            logger.error(f"Error loading video from {video_path}: {e}")
            raise ValueError(f"Error loading video from {video_path}: {e}") from e

    def _tokenize_captions(
        self, captions: Union[str, List[str], Dict[str, Union[str, List[str]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize captions.

        Args:
            captions: Captions to tokenize.

        Returns:
            Dict[str, torch.Tensor]: Tokenized captions.

        Raises:
            ValueError: If tokenization fails.
        """
        try:
            if isinstance(captions, str):
                captions = [captions]
            elif isinstance(captions, dict):
                captions = [v for v in captions.values() if isinstance(v, (str, list))]
            tokenized = self.tokenizer(
                captions,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors='pt',
            )
            return tokenized
        except Exception as e:
            logger.error(f"Error tokenizing captions: {e}")
            raise ValueError(f"Error tokenizing captions: {e}") from e

    def batch_encode(self, texts: List[str], *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Batch encode texts.

        Args:
            texts (List[str]): Texts to encode.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: Encoded texts.
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors='pt',
            *args,
            **kwargs,
        )

    def batch_decode(self, token_ids: torch.Tensor, *args, **kwargs) -> List[str]:
        """
        Batch decode token IDs.

        Args:
            token_ids (torch.Tensor): Token IDs to decode.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: Decoded texts.
        """
        return self.tokenizer.batch_decode(token_ids, *args, **kwargs)


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[Any] = None,
    batch_sampler: Optional[Any] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    *args,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with the specified parameters.

    Args:
        dataset (Dataset): Dataset to load.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle data.
        sampler: Sampler instance.
        batch_sampler: BatchSampler instance.
        num_workers (int): Number of worker threads.
        collate_fn (Optional[Callable]): Collate function.
        pin_memory (bool): Pin memory.
        drop_last (bool): Drop last batch if incomplete.
        timeout (float): Timeout for collecting batch.
        worker_init_fn (Optional[Callable]): Worker init function.
        prefetch_factor (int): Number of samples prefetched.
        persistent_workers (bool): Keep workers alive.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        DataLoader: Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        *args,
        **kwargs,
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-sized data.

    Args:
        batch (List[Dict[str, Any]]): List of samples.

    Returns:
        Dict[str, Any]: Batch with padded sequences.
    """
    videos = [item['video'] for item in batch]
    captions = [item['captions'] for item in batch]
    additional_keys = set(batch[0].keys()) - {'video', 'captions'}
    additional_data = {key: [item[key] for item in batch] for key in additional_keys}

    # Pad videos (assuming videos are tensors of shape [T, H, W, C])
    padded_videos = torch.nn.utils.rnn.pad_sequence(
        [v.permute(0, 3, 1, 2) for v in videos], batch_first=True
    )  # Output shape: [B, T, C, H, W]

    # Pad captions
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [c['input_ids'].squeeze(0) for c in captions], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [c['attention_mask'].squeeze(0) for c in captions], batch_first=True, padding_value=0
    )

    merged_captions = {'input_ids': input_ids, 'attention_mask': attention_mask}

    return {'video': padded_videos, 'captions': merged_captions, **additional_data}


# Example usage:

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Example data
data_entries = [
    {
        'video_path': '/path/to/video1.mp4',
        'captions': 'A cat sitting on the mat.',
        'additional_info': 'Sample 1',
    },
    {
        'video_path': 'https://example.com/video2.mp4',
        'captions': ['A dog playing with a ball.', 'The dog is happy.'],
        'additional_info': 'Sample 2',
    },
]

# Create dataset
dataset = GeneralizedVideoCaptionDataset(
    data=data_entries,
    video_column='video_path',
    caption_column='captions',
    tokenizer=tokenizer,
    max_length=128,
    padding='max_length',
    special_tokens={'eos_token': '[EOS]'},
)

# Create DataLoader
data_loader = create_data_loader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False,
    prefetch_factor=2,
    persistent_workers=True,
)


import os
import urllib.request
import tempfile
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_video
from transformers import AutoTokenizer, BatchEncoding


class VideoCaptionDataset(Dataset):
    """
    A custom Dataset for loading videos and their corresponding captions.
    Supports videos from local paths or URLs and handles arbitrary columns
    with data of type str, List[str], or Dict[str, List[str]].
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer_name_or_path: str,
        transforms: Optional[transforms.Compose] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the VideoCaptionDataset.

        Args:
            data (List[Dict[str, Any]]): List of data samples. Each sample is a dict with arbitrary keys.
            tokenizer_name_or_path (str): Path or name of the tokenizer.
            transforms (Optional[transforms.Compose]): Transformations to be applied on the video data.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Arguments for the tokenizer.
        """
        self.data = data
        self.transforms = transforms
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        except Exception as e:
            raise ValueError(f"Error loading tokenizer: {e}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        result: Dict[str, Any] = {}

        # Handle video loading
        video_path = sample.get("video_path")
        if video_path:
            video = self.load_video(video_path)
            if self.transforms:
                video = self.transforms(video)
            result["video"] = video

        # Handle captions and tokenize them
        caption = sample.get("caption")
        if caption:
            tokens = self.tokenizer(caption, **self.tokenizer_kwargs)
            result["caption_tokens"] = tokens

        # Include other columns
        for key, value in sample.items():
            if key not in ["video_path", "caption"]:
                result[key] = value

        return result

    def load_video(self, path_or_url: str) -> torch.Tensor:
        """
        Loads a video from a local path or URL.

        Args:
            path_or_url (str): The path or URL to the video file.

        Returns:
            torch.Tensor: The video tensor.

        Raises:
            IOError: If the video cannot be loaded.
        """
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            # Download the video to a temporary file
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    urllib.request.urlretrieve(path_or_url, tmp_file.name)
                    video_tensor, _, _ = read_video(tmp_file.name)
                os.unlink(tmp_file.name)
            except Exception as e:
                raise IOError(f"Error loading video from URL {path_or_url}: {e}")
        else:
            if not os.path.exists(path_or_url):
                raise FileNotFoundError(f"Video file not found at the path: {path_or_url}")
            try:
                video_tensor, _, _ = read_video(path_or_url)
            except Exception as e:
                raise IOError(f"Error loading video from path {path_or_url}: {e}")
        return video_tensor


def batch_encode(
    tokenizer: AutoTokenizer,
    texts: List[str],
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> BatchEncoding:
    """
    Performs batch encoding of texts.

    Args:
        tokenizer (AutoTokenizer): The tokenizer instance.
        texts (List[str]): List of texts to be encoded.
        tokenizer_kwargs (Optional[Dict[str, Any]]): Additional arguments for tokenizer.

    Returns:
        BatchEncoding: The encoded batch.
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    return tokenizer(texts, **tokenizer_kwargs)


def batch_decode(
    tokenizer: AutoTokenizer,
    token_ids: List[List[int]],
    decode_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Performs batch decoding of token IDs.

    Args:
        tokenizer (AutoTokenizer): The tokenizer instance.
        token_ids (List[List[int]]): List of token ID sequences.
        decode_kwargs (Optional[Dict[str, Any]]): Additional arguments for decoding.

    Returns:
        List[str]: List of decoded texts.
    """
    decode_kwargs = decode_kwargs or {}
    return [tokenizer.decode(ids, **decode_kwargs) for ids in token_ids]


def create_dataloader(
    datasets: List[Dataset],
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[Any] = None,
    batch_sampler: Optional[Any] = None,
    num_workers: int = 0,
    collate_fn: Optional[Any] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[Any] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """
    Creates a DataLoader from a list of datasets, possibly concatenated or chained.

    Args:
        datasets (List[Dataset]): List of Dataset objects to be combined.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        sampler (Optional[Any]): Defines the strategy to draw samples from the dataset.
        batch_sampler (Optional[Any]): Like sampler, but returns a batch of indices at a time.
        num_workers (int): How many subprocesses to use for data loading.
        collate_fn (Optional[Any]): Merges a list of samples to form a mini-batch.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        drop_last (bool): Set to True to drop the last incomplete batch.
        timeout (float): If positive, the timeout value for collecting a batch from workers.
        worker_init_fn (Optional[Any]): If not None, this will be called on each worker subprocess with the worker id.
        prefetch_factor (int): Number of samples loaded in advance by each worker.
        persistent_workers (bool): If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        DataLoader: A DataLoader object.
    """
    if not datasets:
        raise ValueError("No datasets provided to create_dataloader.")

    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        **kwargs,
    )