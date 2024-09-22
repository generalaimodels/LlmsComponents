import os
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union, Iterator

import requests
import torch
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset
from torchvision import io as video_io
from transformers import AutoTokenizer


class GeneralizedVideoDataset(Dataset):
    """
    A generalized dataset loader for videos and captions.
    Supports any splits, any number of columns, and flexible column data types.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        video_column: str,
        caption_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        special_tokens: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            data: The data source, a list of dictionaries containing dataset samples.
            video_column: The key for video paths in the data.
            caption_column: The key for captions in the data.
            transform: Optional transform to be applied on a sample.
            tokenizer_name: Name of the tokenizer to use from Hugging Face.
            tokenizer_kwargs: Additional kwargs for the tokenizer initialization.
            max_length: Max length for tokenization.
            padding: Padding strategy for tokenization.
            special_tokens: Dict of special tokens to add to the tokenizer.
            **kwargs: Additional arguments.
        """
        self.data = data
        self.video_column = video_column
        self.caption_column = caption_column
        self.transform = transform
        self.tokenizer = None

        if tokenizer_name:
            tokenizer_args = tokenizer_kwargs if tokenizer_kwargs else {}
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_args)

            if special_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})

            self.max_length = max_length
            self.padding = padding

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the sample at the given index.

        Args:
            idx: Index of the sample.

        Returns:
            A dictionary containing the sample data.
        """
        sample = self.data[idx]
        item: Dict[str, Any] = {}

        try:
            video_path = sample[self.video_column]
            video = self._load_video(video_path)
            item['video'] = video

            if self.caption_column:
                caption = sample[self.caption_column]
                if self.tokenizer:
                    encoding = self.tokenizer.encode_plus(
                        caption,
                        max_length=self.max_length,
                        padding=self.padding,
                        truncation=True,
                        return_tensors='pt',
                    )
                    item['caption'] = encoding
                else:
                    item['caption'] = caption

            # Include other columns
            for key, value in sample.items():
                if key not in [self.video_column, self.caption_column]:
                    item[key] = value

            # Apply transform if any
            if self.transform:
                item['video'] = self.transform(item['video'])

        except KeyError as e:
            raise KeyError(f"Missing key in data sample: {e}")

        except Exception as e:
            raise RuntimeError(f"Error loading sample idx {idx}: {e}")

        return item

    def _load_video(self, path_or_url: str) -> torch.Tensor:
        """
        Loads a video from a local path or a URL.

        Args:
            path_or_url: The video file path or URL.

        Returns:
            Video data as a PyTorch tensor.
        """
        try:
            if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
                # URL case
                response = requests.get(path_or_url, stream=True)
                response.raise_for_status()
                video_bytes = BytesIO(response.content)
                video, _, _ = video_io.read_video(video_bytes)
            else:
                # Local file case
                if not os.path.exists(path_or_url):
                    raise FileNotFoundError(f"Video file not found: {path_or_url}")
                video, _, _ = video_io.read_video(path_or_url)
            return video

        except Exception as e:
            raise ValueError(f"Error loading video from {path_or_url}: {e}")

    def batch_encode_captions(self, captions: List[str], **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Batch encodes a list of captions.

        Args:
            captions: A list of caption strings.
            **kwargs: Additional arguments for the tokenizer.

        Returns:
            A dictionary containing the encoded captions.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been initialized.")

        return self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors='pt',
            **kwargs,
        )

    def batch_decode_captions(self, token_ids: torch.Tensor, **kwargs: Any) -> List[str]:
        """
        Batch decodes token IDs to captions.

        Args:
            token_ids: A tensor of token IDs.
            **kwargs: Additional arguments for the tokenizer.

        Returns:
            A list of decoded caption strings.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been initialized.")

        return self.tokenizer.batch_decode(token_ids, **kwargs)


def create_combined_dataset(datasets: List[Dataset]) -> Dataset:
    """
    Combines multiple datasets into a single dataset.

    Args:
        datasets: A list of Dataset objects.

    Returns:
        A combined Dataset.
    """
    if not datasets:
        raise ValueError("No datasets provided for combination.")

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)


def get_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[torch.utils.data.Sampler] = None,
    batch_sampler: Optional[torch.utils.data.BatchSampler] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0.0,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset: The dataset to load data from.
        batch_size: How many samples per batch to load.
        shuffle: Whether to shuffle the data at every epoch.
        sampler: Defines the strategy to draw samples from the dataset.
        batch_sampler: Like sampler, but returns a batch of indices at a time.
        num_workers: How many subprocesses to use for data loading.
        collate_fn: Merges a list of samples to form a mini-batch.
        pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them.
        drop_last: Set to True to drop the last incomplete batch.
        timeout: Timeout value for collecting a batch.
        worker_init_fn: If not None, this will be called on each worker subprocess with the worker id.
        prefetch_factor: Number of batches loaded in advance by each worker.
        persistent_workers: If True, workers are not shut down after dataset iteration.
        **kwargs: Additional arguments.

    Returns:
        A DataLoader instance.
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
        **kwargs,
    )


import os
import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, ChainDataset
from torchvision import transforms, io

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCaptionDataset(Dataset):
    """Generalized dataset loader for videos and captions."""

    def __init__(
        self,
        dataset_name_or_path: Union[str, DatasetDict],
        split: Optional[str] = None,
        columns: Optional[List[str]] = None,
        video_column: str = "video",
        caption_column: str = "caption",
        transform: Optional[Any] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_name_or_path (Union[str, DatasetDict]): The name or path of the dataset.
            split (Optional[str]): The split to use (e.g., 'train', 'validation').
            columns (Optional[List[str]]): List of columns to load.
            video_column (str): Name of the column containing video paths or URLs.
            caption_column (str): Name of the column containing captions.
            transform (Optional[Any]): Transformations to apply to the video data.
            *args: Additional arguments to pass to `load_dataset`.
            **kwargs: Additional keyword arguments to pass to `load_dataset`.
        """
        self.dataset = self.load_dataset(
            dataset_name_or_path, split=split, *args, **kwargs
        )
        self.columns = columns
        self.video_column = video_column
        self.caption_column = caption_column
        self.transform = transform

        # Validate columns
        if self.columns:
            missing_columns = [
                col for col in self.columns if col not in self.dataset.features
            ]
            if missing_columns:
                raise ValueError(f"Missing columns in dataset: {missing_columns}")
        else:
            self.columns = list(self.dataset.features.keys())

    @staticmethod
    def load_dataset(
        name_or_path: Union[str, DatasetDict],
        split: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Load the dataset using the datasets module."""
        try:
            if isinstance(name_or_path, DatasetDict):
                dataset = name_or_path[split] if split else name_or_path
            else:
                dataset = load_dataset(name_or_path, split=split, *args, **kwargs)
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.dataset[idx]
            # Extract video path or URL
            video_data = item.get(self.video_column)
            # Load video frames
            video = self.load_video(video_data)
            # Get caption or other columns
            caption = item.get(self.caption_column, "")
            # Apply transform if provided
            if self.transform and video is not None:
                video = self.transform(video)
            # Prepare the data to return
            data = {"video": video, "caption": caption}
            # Include other columns
            for col in self.columns:
                if col not in [self.video_column, self.caption_column]:
                    data[col] = item.get(col)
            return data
        except Exception as e:
            logger.error(f"Error fetching item at index {idx}: {e}")
            raise e

    def load_video(self, video_data: Union[str, Any]) -> Any:
        """
        Load video from path or URL.

        Args:
            video_data (Union[str, Any]): The video path or data.

        Returns:
            Any: The video data.

        Raises:
            Exception: If video cannot be loaded.
        """
        try:
            if isinstance(video_data, str):
                # Handle URL or local path
                video_path = self.get_video_path(video_data)
                # Load video using torchvision.io
                video, _, _ = io.read_video(video_path)
                return video
            elif isinstance(video_data, bytes):
                # Handle raw video bytes
                import io as sys_io

                video_stream = sys_io.BytesIO(video_data)
                video, _, _ = io.read_video(video_stream)
                return video
            else:
                # Video data is already a tensor or in an unsupported format
                return video_data
        except Exception as e:
            logger.error(f"Error loading video data: {e}")
            return None

    def get_video_path(self, video_data: str) -> str:
        """
        Retrieve the video file path, downloading if necessary.

        Args:
            video_data (str): The video URL or local path.

        Returns:
            str: The local file path to the video.

        Raises:
            Exception: If the video cannot be retrieved.
        """
        try:
            if video_data.startswith(("http://", "https://")):
                # Download the video and save to a temporary location
                return self.download_video(video_data)
            else:
                # Local file path
                if not os.path.exists(video_data):
                    raise FileNotFoundError(f"Video file not found: {video_data}")
                return video_data
        except Exception as e:
            logger.error(f"Error retrieving video path: {e}")
            raise e

    def download_video(self, url: str) -> str:
        """
        Download video from URL to a temporary file.

        Args:
            url (str): The URL of the video.

        Returns:
            str: The path to the downloaded video.

        Raises:
            Exception: If download fails.
        """
        try:
            import tempfile
            import requests

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(url)[-1]
            ) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
            return temp_path
        except Exception as e:
            logger.error(f"Error downloading video from URL {url}: {e}")
            raise e


class RobustTokenizer:
    """Robust tokenizer using transformers AutoTokenizer."""

    def __init__(
        self,
        model_name_or_path: str,
        use_fast: bool = True,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        additional_special_tokens: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: Union[bool, str] = True,
        *args,
        **kwargs,
    ):
        """
        Initialize the tokenizer.

        Args:
            model_name_or_path (str): Name or path of the pre-trained model.
            use_fast (bool): Whether to use the fast tokenizer.
            bos_token (Optional[str]): Beginning-of-sentence token.
            eos_token (Optional[str]): End-of-sentence token.
            additional_special_tokens (Optional[List[str]]): Additional special tokens.
            max_length (Optional[int]): Maximum sequence length.
            padding (Union[bool, str]): Padding strategy.
            truncation (Union[bool, str]): Truncation strategy.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, use_fast=use_fast, *args, **kwargs
            )
            # Set special tokens
            if bos_token:
                self.tokenizer.bos_token = bos_token
            if eos_token:
                self.tokenizer.eos_token = eos_token
            if additional_special_tokens:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": additional_special_tokens}
                )
            self.max_length = max_length
            self.padding = padding
            self.truncation = truncation
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise e

    def encode(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[int], List[List[int]]]:
        """Encode text into token IDs."""
        try:
            return self.tokenizer.encode(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise e

    def decode(
        self, token_ids: Union[List[int], List[List[int]]], **kwargs
    ) -> Union[str, List[str]]:
        """Decode token IDs back to text."""
        try:
            return self.tokenizer.decode(token_ids, **kwargs)
        except Exception as e:
            logger.error(f"Error decoding token IDs: {e}")
            raise e

    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Batch encode a list of texts."""
        try:
            return self.tokenizer.batch_encode_plus(
                texts,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt",
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error batch encoding texts: {e}")
            raise e

    def batch_decode(
        self, token_ids_list: List[List[int]], **kwargs
    ) -> List[str]:
        """Batch decode a list of token ID lists."""
        try:
            return self.tokenizer.batch_decode(token_ids_list, **kwargs)
        except Exception as e:
            logger.error(f"Error batch decoding token IDs: {e}")
            raise e


def create_mixed_dataset(datasets: List[Dataset]) -> Dataset:
    """Create a concatenated dataset from a list of datasets.

    Args:
        datasets (List[Dataset]): List of torch.utils.data.Dataset objects.

    Returns:
        Dataset: The concatenated dataset.
    """
    try:
        mixed_dataset = ConcatDataset(datasets)
        return mixed_dataset
    except Exception as e:
        logger.error(f"Error creating mixed dataset: {e}")
        raise e


def create_chained_dataset(datasets: List[Dataset]) -> Dataset:
    """Create a chained dataset from a list of datasets.

    Args:
        datasets (List[Dataset]): List of torch.utils.data.Dataset objects.

    Returns:
        Dataset: The chained dataset.
    """
    try:
        chained_dataset = ChainDataset(datasets)
        return chained_dataset
    except Exception as e:
        logger.error(f"Error creating chained dataset: {e}")
        raise e


def create_dataloader(
    dataset: Dataset,
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
    *args,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for the dataset.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        sampler (Optional[Any]): Defines the strategy to draw samples from the dataset.
        batch_sampler (Optional[Any]): Like sampler, but returns a batch of indices at a time.
        num_workers (int): How many subprocesses to use for data loading.
        collate_fn (Optional[Any]): Merges a list of samples to form a mini-batch.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory.
        drop_last (bool): Set to True to drop the last incomplete batch.
        timeout (float): Timeout value for collecting a batch.
        worker_init_fn (Optional[Any]): If not None, this will be called on each worker subprocess.
        prefetch_factor (int): Number of batches loaded in advance by each worker.
        persistent_workers (bool): If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        DataLoader: DataLoader for the given dataset.
    """
    try:
        dataloader = DataLoader(
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
        return dataloader
    except Exception as e:
        logger.error(f"Error creating DataLoader: {e}")
        raise e
    

# Initialize dataset
dataset = VideoCaptionDataset(
    dataset_name_or_path='your_dataset_name_or_path',
    split='train',
    columns=['video', 'caption', 'metadata'],
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
)

# Initialize tokenizer
tokenizer = RobustTokenizer(
    model_name_or_path='bert-base-uncased',
    bos_token='[BOS]',
    eos_token='[EOS]',
    additional_special_tokens=['[SPECIAL1]', '[SPECIAL2]'],
    max_length=128,
    padding='max_length',
    truncation=True,
)

# Encode captions
for data in dataset:
    tokens = tokenizer.encode(data['caption'])
    # Continue with training logic

# Create DataLoader
dataloader = create_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# Iterate through DataLoader
for batch in dataloader:
    videos = batch['video']
    captions = batch['caption']
    # Continue with training logic