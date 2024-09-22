import os
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from PIL import Image
from urllib.request import urlopen
from io import BytesIO
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    ChainDataset
)
from torchvision import transforms


class GeneralizedImageCaptionDataset(Dataset):
    """
    A generalized dataset loader for images and captions, supporting various data formats,
    robust tokenization, and extensive customization options.

    Attributes:
        dataset (DatasetDict): The loaded dataset.
        columns (List[str]): Columns to extract from the dataset.
        tokenizer (AutoTokenizer): Tokenizer for text data.
        max_length (Optional[int]): Maximum tokenization length.
        padding (Union[bool, str]): Padding strategy for tokenization.
        transform (transforms.Compose): Transformations to apply to images.
    """

    def __init__(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        columns: Optional[List[str]] = None,
        tokenizer_name: Optional[str] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        special_tokens: Optional[Dict[str, str]] = None,
        transform: Optional[transforms.Compose] = None,
        *args,
        **kwargs
    ) -> None:
        """
        Initializes the GeneralizedImageCaptionDataset.

        Parameters:
            dataset_name (str): Name or path of the dataset.
            split (Optional[str]): Dataset split to load.
            columns (Optional[List[str]]): Columns to extract.
            tokenizer_name (Optional[str]): Name of the tokenizer.
            max_length (Optional[int]): Max tokenization length.
            padding (Union[bool, str]): Padding strategy.
            special_tokens (Optional[Dict[str, str]]): Special tokens to add.
            transform (Optional[transforms.Compose]): Image transformations.
            *args, **kwargs: Additional arguments for dataset loading.
        """
        try:
            self.dataset = load_dataset(dataset_name, split=split, *args, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_name}' with split '{split}': {e}")

        self.columns = columns if columns else self.dataset.column_names
        self.tokenizer = None

        if tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                if special_tokens:
                    self.tokenizer.add_special_tokens(special_tokens)
            except Exception as e:
                raise ValueError(f"Failed to load tokenizer '{tokenizer_name}': {e}")

        self.max_length = max_length
        self.padding = padding
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        data = {}

        for column in self.columns:
            value = sample.get(column)
            if value is None:
                continue

            # Process image columns
            if self.is_image_data(value):
                try:
                    image = self.load_image(value)
                    image = self.transform(image)
                    data[column] = image
                except Exception as e:
                    raise RuntimeError(f"Error processing image in column '{column}': {e}")

            # Process text columns
            elif isinstance(value, (str, list, dict)):
                if self.tokenizer:
                    try:
                        tokens = self.tokenize_text(value)
                        data[f"{column}_input_ids"] = tokens['input_ids']
                        data[f"{column}_attention_mask"] = tokens['attention_mask']
                    except Exception as e:
                        raise RuntimeError(f"Error tokenizing text in column '{column}': {e}")
                else:
                    data[column] = value

            # Process other data types
            else:
                data[column] = value

        return data

    def is_image_data(self, value: Any) -> bool:
        """
        Determines if the provided value corresponds to image data.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if value is image data, False otherwise.
        """
        if isinstance(value, str):
            return value.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) or value.startswith(('http://', 'https://'))
        elif isinstance(value, bytes):
            return True
        else:
            return False

    def load_image(self, image_data: Union[str, bytes]) -> Image.Image:
        """
        Loads an image from a file path, URL, or raw bytes.

        Parameters:
            image_data (Union[str, bytes]): The image data to load.

        Returns:
            Image.Image: The loaded PIL Image.
        """
        try:
            if isinstance(image_data, str):
                if image_data.startswith(('http://', 'https://')):
                    with urlopen(image_data) as response:
                        image = Image.open(BytesIO(response.read()))
                else:
                    image = Image.open(image_data)
            elif isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data))
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
            return image.convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load image: {e}")

    def tokenize_text(self, text: Union[str, List[str], Dict[str, List[str]]]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes text data using the configured tokenizer.

        Parameters:
            text (Union[str, List[str], Dict[str, List[str]]]): The text data to tokenize.

        Returns:
            Dict[str, torch.Tensor]: The tokenized outputs.
        """
        if isinstance(text, str):
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors='pt'
            )
        elif isinstance(text, list):
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors='pt'
            )
        elif isinstance(text, dict):
            tokens = {}
            for key, value in text.items():
                tokens[key] = self.tokenizer(
                    value,
                    max_length=self.max_length,
                    padding=self.padding,
                    truncation=True,
                    return_tensors='pt'
                )
        else:
            raise ValueError(f"Unsupported text data type: {type(text)}")
        return tokens


# Example usage:

if __name__ == "__main__":
    # Define special tokens if needed
    special_tokens_dict = {
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
        'sep_token': '<SEP>',
        'pad_token': '<PAD>'
    }

    # Initialize the dataset
    dataset = GeneralizedImageCaptionDataset(
        dataset_name='coco_captions',
        split='train',
        columns=['image', 'captions'],
        tokenizer_name='bert-base-uncased',
        max_length=128,
        padding='max_length',
        special_tokens=special_tokens_dict
    )

    # Initialize another dataset (e.g., for mixing datasets)
    another_dataset = GeneralizedImageCaptionDataset(
        dataset_name='flickr30k',
        split='train',
        columns=['image', 'caption'],
        tokenizer_name='bert-base-uncased',
        max_length=128,
        padding='max_length',
        special_tokens=special_tokens_dict
    )

    # Mixing datasets using ConcatDataset
    mixed_dataset = ConcatDataset([dataset, another_dataset])

    # Creating DataLoader
    dataloader = DataLoader(
        mixed_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x,  # Customize collate_fn as needed
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Iterating through the DataLoader
    for batch in dataloader:
        # Process the batch
        # Example: Access images and captions
        images = [item['image'] for item in batch]
        captions_input_ids = [item['captions_input_ids'] for item in batch if 'captions_input_ids' in item]
        captions_attention_masks = [item['captions_attention_mask'] for item in batch if 'captions_attention_mask' in item]

        # Continue with training loop or processing
        pass

    # Example of extending datasets for day-to-day training using ChainDataset
    daily_dataset_1 = GeneralizedImageCaptionDataset(
        dataset_name='daily_dataset_day1',
        columns=['image', 'text'],
        tokenizer_name='bert-base-uncased'
    )

    daily_dataset_2 = GeneralizedImageCaptionDataset(
        dataset_name='daily_dataset_day2',
        columns=['image', 'text'],
        tokenizer_name='bert-base-uncased'
    )

    continuous_dataset = ChainDataset([daily_dataset_1, daily_dataset_2])

    continuous_dataloader = DataLoader(
        continuous_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: x,
        pin_memory=False,
        drop_last=False
    )

    # Iterate over the continuous dataset
    for batch in continuous_dataloader:
        # Process the batch for day-to-day training
        pass


import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
import torch
from PIL import Image
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class GeneralizedImageCaptionDataset(Dataset):
    """
    A generalized dataset loader for images and captions that handles any splits,
    any number of columns, and various data formats, including strings, lists of strings,
    and dictionaries mapping strings to lists of strings. Images can be specified by absolute paths
    or URLs. All columns are properly loaded.
    """

    def __init__(
        self,
        data: Sequence[Dict[str, Any]],
        image_column: str,
        columns: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data (Sequence[Dict[str, Any]]): The dataset represented as a sequence of dictionaries.
            image_column (str): The key in the data dictionaries that corresponds to the image path or URL.
            columns (Optional[List[str]], optional): List of columns to include. If None, include all columns. Defaults to None.
            transform (Optional[transforms.Compose], optional): Transformations to apply to the images. Defaults to None.
            *args, **kwargs: Additional arguments.
        """
        self.data = data
        self.image_column = image_column
        self.columns = columns
        self.transform = transform
        self.args = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, image_path_or_url: str) -> Image.Image:
        try:
            if image_path_or_url.startswith("http://") or image_path_or_url.startswith(
                "https://"
            ):
                # Load image from URL
                response = requests.get(image_path_or_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Load image from local file system
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(
                        f"Image file not found: {image_path_or_url}"
                    )
                image = Image.open(image_path_or_url).convert("RGB")
            return image
        except Exception as e:
            raise IOError(f"Error loading image from {image_path_or_url}: {e}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        # Load the image
        image_path_or_url = item.get(self.image_column)
        if image_path_or_url is None:
            raise KeyError(
                f"Image column '{self.image_column}' not found in data at index {idx}."
            )

        image = self._load_image(image_path_or_url)

        if self.transform:
            image = self.transform(image)

        # Prepare the result
        result = {"image": image}

        # Process other columns
        if self.columns is not None:
            column_keys = self.columns
        else:
            column_keys = [key for key in item.keys() if key != self.image_column]

        for key in column_keys:
            value = item.get(key)
            # Include all value types
            result[key] = value

        return result


class RobustTokenizer:
    """
    A wrapper around transformers.AutoTokenizer to provide robust tokenization
    capabilities, including support for special tokens, max_length, padding,
    batch encoding and decoding.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        use_special_tokens: bool = True,
        add_bos_token: Optional[bool] = None,
        add_eos_token: Optional[bool] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the tokenizer.

        Args:
            tokenizer_name (str): Name or path of the tokenizer.
            max_length (Optional[int], optional): Maximum length of the sequences. Defaults to None.
            padding (Union[bool, str], optional): Padding strategy. Defaults to False.
            truncation (Union[bool, str], optional): Truncation strategy. Defaults to False.
            use_special_tokens (bool, optional): Whether to use special tokens. Defaults to True.
            add_bos_token (Optional[bool], optional): Whether to add BOS token at the beginning. Defaults to None.
            add_eos_token (Optional[bool], optional): Whether to add EOS token at the end. Defaults to None.
            *args, **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, *args, **kwargs)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.use_special_tokens = use_special_tokens
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def encode(
        self,
        text: Union[str, List[str]],
        **kwargs: Any
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text into token ids.

        Args:
            text (Union[str, List[str]]): Text or list of texts to encode.
            **kwargs: Additional arguments passed to tokenizer.encode or tokenizer.batch_encode_plus.

        Returns:
            Union[List[int], List[List[int]]]: Token ids or list of token ids.
        """
        if isinstance(text, str):
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                add_special_tokens=self.use_special_tokens,
                **kwargs
            )
            if self.add_bos_token:
                tokens = [self.tokenizer.bos_token_id] + tokens
            if self.add_eos_token:
                tokens = tokens + [self.tokenizer.eos_token_id]
            return tokens
        elif isinstance(text, list):
            encoded = self.tokenizer.batch_encode_plus(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                add_special_tokens=self.use_special_tokens,
                **kwargs
            )
            input_ids = encoded["input_ids"]
            if self.add_bos_token or self.add_eos_token:
                modified_input_ids = []
                for ids in input_ids:
                    if self.add_bos_token:
                        ids = [self.tokenizer.bos_token_id] + ids
                    if self.add_eos_token:
                        ids = ids + [self.tokenizer.eos_token_id]
                    modified_input_ids.append(ids)
                input_ids = modified_input_ids
            return input_ids
        else:
            raise TypeError(
                f"Input text must be a string or list of strings, got {type(text)}."
            )

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        **kwargs: Any
    ) -> Union[str, List[str]]:
        """
        Decode token ids back into text.

        Args:
            token_ids (Union[List[int], List[List[int]]]): Token ids to decode.
            **kwargs: Additional arguments passed to tokenizer.decode or tokenizer.batch_decode.

        Returns:
            Union[str, List[str]]: Decoded text or list of texts.
        """
        if isinstance(token_ids[0], int):
            # Single sequence
            text = self.tokenizer.decode(token_ids, **kwargs)
            return text
        elif isinstance(token_ids[0], list):
            # Batch sequences
            texts = self.tokenizer.batch_decode(token_ids, **kwargs)
            return texts
        else:
            raise TypeError(
                f"Token IDs must be a list of ints or list of lists of ints, got {type(token_ids)}."
            )


def create_data_loader(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: Optional[int] = 1,
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
    use_concat: bool = False,
    *args: Any,
    **kwargs: Any
) -> DataLoader:
    """
    Create a DataLoader from one or multiple datasets, optionally concatenating or chaining them.

    Args:
        datasets (Union[Dataset, List[Dataset]]): Single dataset or list of datasets.
        batch_size (Optional[int], optional): Batch size. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle data. Defaults to False.
        sampler (Optional[Any], optional): Sampler. Defaults to None.
        batch_sampler (Optional[Any], optional): Batch sampler. Defaults to None.
        num_workers (int, optional): Number of worker processes. Defaults to 0.
        collate_fn (Optional[Any], optional): Collate function. Defaults to None.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        timeout (float, optional): Timeout for collecting a batch. Defaults to 0.
        worker_init_fn (Optional[Any], optional): Worker initialization function. Defaults to None.
        prefetch_factor (int, optional): Number of batches loaded in advance. Defaults to 2.
        persistent_workers (bool, optional): Whether to keep workers persistent. Defaults to False.
        use_concat (bool, optional): Whether to concatenate the datasets (True) or chain them (False). Defaults to False.
        *args, **kwargs: Additional arguments passed to DataLoader.

    Returns:
        DataLoader: The data loader.
    """
    if isinstance(datasets, Dataset):
        dataset = datasets
    elif isinstance(datasets, list):
        if use_concat:
            dataset = ConcatDataset(datasets)
        else:
            dataset = ChainDataset(datasets)
    else:
        raise TypeError(
            f"Datasets must be a Dataset or list of Datasets, got {type(datasets)}."
        )

    data_loader = DataLoader(
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
        **kwargs
    )

    return data_loader



# from torchvision import transforms

# # Define transformations for images
# image_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # Sample data
# data = [
#     {
#         'image_path': '/path/to/image1.jpg',
#         'caption': 'A caption for image 1.',
#         'metadata': {'id': 1, 'tags': ['tag1', 'tag2']},
#     },
#     {
#         'image_path': 'https://example.com/image2.png',
#         'caption': ['Caption part 1', 'Caption part 2'],
#         'metadata': {'id': 2, 'tags': ['tag3']},
#     },
# ]

# # Initialize the dataset
# dataset = GeneralizedImageCaptionDataset(
#     data=data,
#     image_column='image_path',
#     transform=image_transforms
# )

# # Initialize the tokenizer
# tokenizer = RobustTokenizer(
#     tokenizer_name='bert-base-uncased',
#     max_length=128,
#     padding='max_length',
#     truncation=True,
#     add_eos_token=True
# )

# # Create a data loader
# data_loader = create_data_loader(
#     dataset,
#     batch_size=8,
#     shuffle=True,
#     num_workers=4
# )

# # Iterate over the data loader
# for batch in data_loader:
#     images = batch['image']
#     captions = batch['caption']
#     # Tokenize captions
#     tokenized_captions = tokenizer.encode(captions)
#     # Continue with training...




