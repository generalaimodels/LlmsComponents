import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import requests
from PIL import Image
from torch.utils.data import (
    ChainDataset,
    ConcatDataset,
    DataLoader,
    Dataset,
)
from torchvision import transforms
from transformers import AutoTokenizer, BatchEncoding

# Constants for type hints
SplitType = Optional[str]
ColumnType = Union[str, List[str], Dict[str, List[str]]]
ImageType = Union[Image.Image, None]


def download_image(url: str) -> ImageType:
    """
    Downloads an image from a URL and returns a PIL.Image object.

    Args:
        url (str): URL of the image to download.

    Returns:
        ImageType: PIL.Image object if successful, None otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        warnings.warn(f"Failed to download image from {url}: {e}")
        return None


def load_image(path_or_url: str) -> ImageType:
    """
    Loads an image from a given path or URL.

    Args:
        path_or_url (str): Local file path or URL to the image.

    Returns:
        ImageType: PIL.Image object if successful, None otherwise.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return download_image(path_or_url)
    else:
        try:
            return Image.open(path_or_url).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load image from {path_or_url}: {e}")
            return None


class GeneralizedImageCaptionDataset(Dataset):
    """
    A generalized dataset loader for images and captions that handles various data formats and sources.

    Attributes:
        data (List[Dict[str, Any]]): List of data entries with image and captions.
        image_transform (Optional[Callable]): Transformations to apply to images.
        tokenizer (Optional[AutoTokenizer]): Tokenizer for text data.
        max_length (Optional[int]): Maximum sequence length for tokenization.
        padding (Union[bool, str]): Padding strategy for tokenization.
        additional_args (Dict[str, Any]): Additional arguments for flexibility.
    """

    def __init__(
        self,
        data_entries: Sequence[Dict[str, Any]],
        image_column: str,
        caption_column: Union[str, List[str]],
        split: SplitType = None,
        image_transform: Optional[Callable] = None,
        tokenizer_name: Optional[str] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = "max_length",
        **additional_args: Any,
    ) -> None:
        """
        Initializes the dataset with data entries and configurations.

        Args:
            data_entries (Sequence[Dict[str, Any]]): Dataset entries containing image paths/URLs and captions.
            image_column (str): Key name for image data in the entries.
            caption_column (Union[str, List[str]]): Key name(s) for caption data in the entries.
            split (SplitType, optional): Dataset split type. Defaults to None.
            image_transform (Optional[Callable], optional): Transformations for images. Defaults to None.
            tokenizer_name (Optional[str], optional): Name of the tokenizer. Defaults to None.
            max_length (Optional[int], optional): Max sequence length for tokenization. Defaults to None.
            padding (Union[bool, str], optional): Padding strategy. Defaults to "max_length".
            **additional_args: Additional keyword arguments.
        """
        self.data = data_entries
        self.image_column = image_column
        self.caption_column = caption_column
        self.split = split
        self.image_transform = image_transform or transforms.ToTensor()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            if tokenizer_name
            else None
        )
        self.max_length = max_length
        self.padding = padding
        self.additional_args = additional_args

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the dataset item at the specified index after processing.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: Processed data item with image and caption (tokenized if tokenizer is provided).
        """
        item = self.data[idx]

        # Load and transform image
        image = load_image(item[self.image_column])
        if image is not None and self.image_transform:
            image = self.image_transform(image)
        else:
            image = None

        # Retrieve and process captions
        captions = item[self.caption_column]
        if isinstance(captions, str):
            captions = [captions]

        # Tokenize captions if tokenizer is available
        tokenized_captions = None
        if self.tokenizer:
            tokenized_captions = self.tokenizer(
                captions,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt",
                **self.additional_args,
            )

        return {
            "image": image,
            "captions": captions,
            "tokenized_captions": tokenized_captions,
            **{k: item.get(k) for k in item if k not in [self.image_column, self.caption_column]},
        }


def collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Union[List[Any], BatchEncoding]]:
    """
    Custom collate function to handle batching of dataset items.

    Args:
        batch (List[Dict[str, Any]]): List of dataset items.

    Returns:
        Dict[str, Union[List[Any], BatchEncoding]]: Batched data.
    """
    images = [item["image"] for item in batch if item["image"] is not None]
    captions = [item["captions"] for item in batch]
    tokenized_captions = (
        batch[0]["tokenized_captions"].__class__.from_batches([item["tokenized_captions"] for item in batch])
        if batch[0]["tokenized_captions"] is not None
        else None
    )

    # Collect other fields
    other_fields = {
        key: [item[key] for item in batch if key in item]
        for key in batch[0]
        if key not in ["image", "captions", "tokenized_captions"]
    }

    return {
        "images": images,
        "captions": captions,
        "tokenized_captions": tokenized_captions,
        **other_fields,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """
    Creates a DataLoader with given configurations.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
        drop_last (bool, optional): Drop last incomplete batch. Defaults to False.
        prefetch_factor (int, optional): Number of batches to prefetch per worker. Defaults to 2.
        persistent_workers (bool, optional): Keep workers alive between epochs. Defaults to False.
        **kwargs: Additional keyword arguments for DataLoader.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        **kwargs,
    )


def mix_datasets(*datasets: Dataset) -> Dataset:
    """
    Concatenates multiple datasets into one.

    Args:
        *datasets (Dataset): Datasets to combine.

    Returns:
        Dataset: Combined dataset.
    """
    return ConcatDataset(datasets)


# Example usage:
if __name__ == "__main__":
    # Sample data entries
    data_entries = [
        {
            "image_path": "/path/to/image1.jpg",
            "caption": "A caption for image 1.",
            "additional_info": {"id": 1},
        },
        {
            "image_path": "https://example.com/image2.jpg",
            "caption": ["Caption line 1 for image 2.", "Caption line 2 for image 2."],
            "additional_info": {"id": 2},
        },
    ]

    # Initialize dataset
    dataset = GeneralizedImageCaptionDataset(
        data_entries=data_entries,
        image_column="image_path",
        caption_column="caption",
        image_transform=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        ),
        tokenizer_name="bert-base-uncased",
        max_length=128,
        padding="max_length",
    )

    # Create DataLoader
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # Iterate over DataLoader
    for batch in dataloader:
        images = batch["images"]
        captions = batch["captions"]
        tokenized_captions = batch["tokenized_captions"]
        # Further processing...


import os
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torchvision import transforms

class GeneralizedImageCaptionDataset(Dataset):
    """
    A generalized Dataset class for loading images and captions with support for various data formats,
    arbitrary columns, and robust handling of image paths or URLs.
    """
    def __init__(
        self,
        data: List[Dict[str, Any]],
        image_key: str,
        caption_key: Union[str, List[str]],
        transform: Optional[Callable] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        add_special_tokens: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initializes the dataset.

        Args:
            data (List[Dict[str, Any]]): The dataset represented as a list of dictionaries.
            image_key (str): The key used to extract image paths or URLs from the data.
            caption_key (Union[str, List[str]]): The key(s) used to extract captions from the data.
            transform (Optional[Callable], optional): Transformations to apply to the images. Defaults to None.
            tokenizer (Optional[PreTrainedTokenizerBase], optional): Tokenizer for processing captions. Defaults to None.
            max_length (Optional[int], optional): Maximum length for tokenized captions. Defaults to None.
            padding (Union[bool, str], optional): Padding strategy for tokenization. Defaults to True.
            add_special_tokens (bool, optional): Whether to add special tokens during tokenization. Defaults to True.
            **kwargs (Any): Additional keyword arguments.
        """
        self.data = data
        self.image_key = image_key
        self.caption_key = caption_key
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.tokenizer_kwargs = kwargs  # Additional tokenizer arguments

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from a local path or URL.

        Args:
            image_path (str): The path or URL to the image.

        Returns:
            Image.Image: The loaded image.
        """
        try:
            if image_path.startswith('http://') or image_path.startswith('https://'):
                response = requests.get(image_path)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                img = Image.open(image_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {e}")

    def _process_caption(self, caption: Union[str, List[str], Dict[str, Any]]) -> Any:
        """
        Processes the caption based on its type using the tokenizer.

        Args:
            caption (Union[str, List[str], Dict[str, Any]]): The caption data.

        Returns:
            Any: The processed caption.
        """
        if self.tokenizer:
            if isinstance(caption, str):
                tokens = self.tokenizer.encode(
                    caption,
                    max_length=self.max_length,
                    padding=self.padding,
                    add_special_tokens=self.add_special_tokens,
                    **self.tokenizer_kwargs
                )
                return tokens
            elif isinstance(caption, list):
                tokens = self.tokenizer.batch_encode_plus(
                    caption,
                    max_length=self.max_length,
                    padding=self.padding,
                    add_special_tokens=self.add_special_tokens,
                    **self.tokenizer_kwargs
                )
                return tokens['input_ids']
            elif isinstance(caption, dict):
                tokens = {k: self._process_caption(v) for k, v in caption.items()}
                return tokens
            else:
                raise ValueError(f"Unsupported caption type: {type(caption)}")
        else:
            return caption

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        sample = {}
        # Load image
        image_path = item.get(self.image_key)
        if not image_path:
            raise KeyError(f"Image key '{self.image_key}' not found in data at index {index}")
        sample['image'] = self._load_image(image_path)
        # Process caption(s)
        if isinstance(self.caption_key, str):
            caption = item.get(self.caption_key)
            if caption is None:
                raise KeyError(f"Caption key '{self.caption_key}' not found in data at index {index}")
            sample['caption'] = self._process_caption(caption)
        elif isinstance(self.caption_key, list):
            captions = {}
            for key in self.caption_key:
                caption = item.get(key)
                if caption is None:
                    raise KeyError(f"Caption key '{key}' not found in data at index {index}")
                captions[key] = self._process_caption(caption)
            sample['captions'] = captions
        else:
            raise ValueError(f"Unsupported caption_key type: {type(self.caption_key)}")
        # Include other columns
        for k, v in item.items():
            if k not in [self.image_key] + ([self.caption_key] if isinstance(self.caption_key, str) else self.caption_key):
                sample[k] = v
        return sample

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs: Any
) -> DataLoader:
    """
    Creates a DataLoader with the specified parameters.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int, optional): How many samples per batch to load. Defaults to 1.
        shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to False.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 0.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory. Defaults to False.
        drop_last (bool, optional): Set to True to drop the last incomplete batch. Defaults to False.
        prefetch_factor (int, optional): Number of batches loaded in advance by each worker. Defaults to 2.
        persistent_workers (bool, optional): If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to False.
        collate_fn (Optional[Callable], optional): Merges a list of samples to form a mini-batch. Defaults to None.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        DataLoader: The configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **kwargs
    )

def create_tokenizer(
    pretrained_model_name_or_path: str,
    use_fast: bool = True,
    add_special_tokens: Optional[Dict[str, str]] = None,
    **kwargs: Any
) -> PreTrainedTokenizerBase:
    """
    Creates a tokenizer with the specified parameters.

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained tokenizer.
        use_fast (bool, optional): Whether to use a fast tokenizer. Defaults to True.
        add_special_tokens (Optional[Dict[str, str]], optional): Special tokens to add to the tokenizer. Defaults to None.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        PreTrainedTokenizerBase: The configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast, **kwargs)
    if add_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': list(add_special_tokens.values())})
    return tokenizer

from transformers import AutoTokenizer
from torchvision import transforms

# Define transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create tokenizer
special_tokens = {'eos_token': '</s>', 'bos_token': '<s>'}
tokenizer = create_tokenizer('bert-base-uncased', add_special_tokens=special_tokens)

# Sample data
data = [
    {
        'image_path': '/path/to/image1.jpg',
        'caption': 'A caption for image one.',
        'additional_info': {'label': 'cat'}
    },
    {
        'image_path': 'https://example.com/image2.png',
        'caption': 'A caption for image two.',
        'additional_info': {'label': 'dog'}
    },
]

# Initialize dataset
dataset = GeneralizedImageCaptionDataset(
    data=data,
    image_key='image_path',
    caption_key='caption',
    transform=image_transforms,
    tokenizer=tokenizer,
    max_length=128,
    padding='max_length',
    add_special_tokens=True
)

# Create DataLoader
dataloader = create_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate through data
for batch in dataloader:
    images = batch['image']
    captions = batch['caption']
    # Training logic here