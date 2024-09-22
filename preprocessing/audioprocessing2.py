import logging
import os
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Union

import librosa
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, ChainDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralDataset(Dataset):
    """A generalized dataset loader for audio and captions."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        transforms: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            data (List[Dict[str, Any]]): List of data entries, each being a dict with keys and values.
            transforms (Optional[Callable]): Function to transform a sample.
            **kwargs: Additional keyword arguments for data loading functions (e.g., librosa.load).
        """
        super().__init__()
        self.data = data
        self.transforms = transforms
        self.kwargs = kwargs

    def __len__(self) -> int:
        return len(self.data)

    def _load_audio(self, audio_source: str) -> torch.Tensor:
        """Load an audio file from a path or URL.

        Args:
            audio_source (str): Path or URL to the audio file.

        Returns:
            torch.Tensor: Loaded audio signal.
        """
        try:
            if audio_source.startswith(("http://", "https://")):
                # Download the audio file to a temporary location
                temp_filename, _ = urllib.request.urlretrieve(audio_source)
                audio, _ = librosa.load(temp_filename, **self.kwargs)
                os.remove(temp_filename)
            else:
                audio, _ = librosa.load(audio_source, **self.kwargs)
            return torch.tensor(audio)
        except Exception as e:
            logger.error(f"Error loading audio from {audio_source}: {e}")
            return torch.tensor([])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        processed_sample = {}
        for key, value in sample.items():
            if key == "audio":
                if isinstance(value, str):
                    processed_sample[key] = self._load_audio(value)
                elif isinstance(value, list):
                    processed_sample[key] = [self._load_audio(v) for v in value]
                elif isinstance(value, dict):
                    processed_sample[key] = {
                        k: self._load_audio(v) for k, v in value.items()
                    }
                else:
                    logger.warning(f"Unsupported audio data type: {type(value)}")
                    processed_sample[key] = value
            else:
                processed_sample[key] = value

        if self.transforms:
            processed_sample = self.transforms(processed_sample)

        return processed_sample


class TokenizerWrapper:
    """A wrapper for the AutoTokenizer with advanced features."""

    def __init__(
        self,
        tokenizer_name_or_path: str,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the tokenizer.

        Args:
            tokenizer_name_or_path (str): Name or path to the tokenizer.
            bos_token (Optional[str]): Beginning of sentence token.
            eos_token (Optional[str]): End of sentence token.
            pad_token (Optional[str]): Padding token.
            additional_special_tokens (Optional[List[str]]): Additional special tokens.
            **kwargs: Additional arguments for the tokenizer.
        """
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, **kwargs
        )

        special_tokens_dict = {}
        if bos_token:
            special_tokens_dict["bos_token"] = bos_token
        if eos_token:
            special_tokens_dict["eos_token"] = eos_token
        if pad_token:
            special_tokens_dict["pad_token"] = pad_token
        if additional_special_tokens:
            special_tokens_dict["additional_special_tokens"] = additional_special_tokens

        if special_tokens_dict:
            self.tokenizer.add_special_tokens(special_tokens_dict)

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text(s) to token IDs.

        Args:
            text (Union[str, List[str]]): Text(s) to encode.
            max_length (Optional[int]): Maximum sequence length.
            padding (Union[bool, str]): Padding strategy.
            truncation (Union[bool, str]): Truncation strategy.
            add_special_tokens (bool): Add special tokens.
            **kwargs: Additional arguments for encoding.

        Returns:
            Union[List[int], List[List[int]]]: Encoded token IDs.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
            **kwargs,
        )
        return encoding["input_ids"]

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: Union[bool, str] = True,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[List[int]]:
        """
        Batch encode texts.

        Args:
            texts (List[str]): Texts to encode.
            max_length (Optional[int]): Maximum sequence length.
            padding (Union[bool, str]): Padding strategy.
            truncation (Union[bool, str]): Truncation strategy.
            add_special_tokens (bool): Add special tokens.
            **kwargs: Additional arguments for encoding.

        Returns:
            List[List[int]]: Encoded token IDs.
        """
        return self.encode(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text(s).

        Args:
            token_ids (Union[List[int], List[List[int]]]): Token IDs to decode.
            skip_special_tokens (bool): Remove special tokens in decoding.
            **kwargs: Additional arguments for decoding.

        Returns:
            Union[str, List[str]]: Decoded text(s).
        """
        if isinstance(token_ids[0], list):
            return self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )
        else:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )

    def batch_decode(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Batch decode token IDs.

        Args:
            token_ids_list (List[List[int]]): List of token IDs.
            skip_special_tokens (bool): Remove special tokens in decoding.
            **kwargs: Additional arguments for decoding.

        Returns:
            List[str]: Decoded texts.
        """
        return self.tokenizer.batch_decode(
            token_ids_list,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )


def get_data_loader(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: int = 1,
    shuffle: bool = False,
    sampler=None,
    batch_sampler=None,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[Callable] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader from datasets.

    Args:
        datasets (Union[Dataset, List[Dataset]]): Dataset(s) to load.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle data.
        sampler: Sampling strategy.
        batch_sampler: Batch sampling strategy.
        num_workers (int): Number of worker processes.
        collate_fn (Optional[Callable]): Function to merge a list of samples.
        pin_memory (bool): Use pinned memory.
        drop_last (bool): Drop last incomplete batch.
        timeout (float): Timeout for collecting a batch.
        worker_init_fn (Optional[Callable]): Function to initialize workers.
        prefetch_factor (int): Number of samples to prefetch.
        persistent_workers (bool): Keep workers alive.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: Configured DataLoader.
    """
    if isinstance(datasets, list):
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            if kwargs.get("use_chain_dataset", False):
                dataset = ChainDataset(datasets)
            else:
                dataset = ConcatDataset(datasets)
    else:
        dataset = datasets

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
from io import BytesIO
from typing import List, Dict, Any, Union, Optional, Callable

import requests
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, ChainDataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralizedAudioCaptionDataset(Dataset):
    """
    Generalized Dataset loader for audio and captions.
    
    This dataset loader can handle any splits, any number of columns, and columns of type str,
    list(str), or dict(str, list(str)). It properly loads all columns, handles audio files from
    absolute paths or URLs, and supports robust tokenization using transformers' AutoTokenizer.
    
    Args:
        dataset_names_or_paths (List[Union[str, Dict[str, Any]]]): List of dataset names or paths.
        tokenizer_name (str): Name of the tokenizer to use.
        split (str, optional): Data split to use ('train', 'validation', 'test', etc.).
        columns (Optional[List[str]], optional): List of columns to load.
        max_length (int, optional): Maximum length of tokenized sequences.
        padding (Union[bool, str], optional): Padding strategy ('max_length', 'longest', etc.).
        special_tokens (Optional[Dict[str, str]], optional): Special tokens to add to the tokenizer.
        transform (Optional[Callable], optional): A function/transform to apply to audio samples.
        *args, **kwargs: Additional arguments.
    """
    def __init__(
        self,
        dataset_names_or_paths: List[Union[str, Dict[str, Any]]],
        tokenizer_name: str,
        split: str = 'train',
        columns: Optional[List[str]] = None,
        max_length: int = 512,
        padding: Union[bool, str] = True,
        special_tokens: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.dataset_names_or_paths = dataset_names_or_paths
        self.split = split
        self.columns = columns
        self.max_length = max_length
        self.padding = padding
        self.transform = transform

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            if special_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
                logger.info(f"Added special tokens: {special_tokens}")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer '{tokenizer_name}': {e}")
            raise

        # Load datasets
        self.datasets = self._load_datasets(*args, **kwargs)
        self.total_length = sum(len(d) for d in self.datasets)

    def _load_datasets(self, *args, **kwargs) -> List[Dataset]:
        """
        Internal method to load multiple datasets.

        Returns:
            List[Dataset]: List of loaded datasets.
        """
        datasets_list = []
        for data_source in self.dataset_names_or_paths:
            try:
                if isinstance(data_source, str):
                    if os.path.exists(data_source):
                        # Load from local files
                        dataset = load_dataset('csv', data_files=data_source, split=self.split, *args, **kwargs)
                        logger.info(f"Loaded dataset from local file: {data_source}")
                    else:
                        # Load from datasets library
                        dataset = load_dataset(data_source, split=self.split, *args, **kwargs)
                        logger.info(f"Loaded dataset: {data_source}")
                elif isinstance(data_source, dict):
                    # Directly use provided data
                    dataset = data_source
                    logger.info("Loaded dataset from provided data dict")
                else:
                    raise TypeError(f"Unsupported data source type: {type(data_source)}")
                
                if self.columns:
                    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in self.columns])
                    logger.info(f"Selected columns: {self.columns}")

                datasets_list.append(dataset)
            except Exception as e:
                logger.error(f"Error loading dataset '{data_source}': {e}")
                continue
        return datasets_list

    def __len__(self) -> int:
        return self.total_length

    def _load_audio(self, path_or_url: str) -> Any:
        """
        Load audio from a local path or a URL.

        Args:
            path_or_url (str): Path to the audio file or URL.

        Returns:
            Any: Audio data.
        """
        try:
            if path_or_url.startswith(('http://', 'https://')):
                response = requests.get(path_or_url)
                response.raise_for_status()
                audio_data, sample_rate = librosa.load(BytesIO(response.content), sr=None)
            else:
                audio_data, sample_rate = librosa.load(path_or_url, sr=None)
            if self.transform:
                audio_data = self.transform(audio_data)
            return audio_data
        except Exception as e:
            logger.error(f"Failed to load audio from '{path_or_url}': {e}")
            raise

    def _process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example.

        Args:
            example (Dict[str, Any]): Raw example.

        Returns:
            Dict[str, Any]: Processed example.
        """
        result = {}
        for key, value in example.items():
            if key == 'audio':
                if isinstance(value, str):
                    result['audio'] = self._load_audio(value)
                elif isinstance(value, list):
                    result['audio'] = [self._load_audio(v) for v in value]
                elif isinstance(value, dict):
                    result['audio'] = {k: [self._load_audio(vv) for vv in v] for k, v in value.items()}
                else:
                    logger.warning(f"Unknown audio data type for key '{key}': {type(value)}")
                    result['audio'] = value
            elif key == 'caption':
                # Tokenize captions
                if isinstance(value, (str, list)):
                    tokens = self.tokenizer(
                        value,
                        max_length=self.max_length,
                        padding=self.padding,
                        truncation=True,
                        return_tensors='pt',
                        add_special_tokens=True
                    )
                    result['input_ids'] = tokens['input_ids']
                    result['attention_mask'] = tokens['attention_mask']
                elif isinstance(value, dict):
                    tokens = {k: self.tokenizer(
                        v,
                        max_length=self.max_length,
                        padding=self.padding,
                        truncation=True,
                        return_tensors='pt',
                        add_special_tokens=True
                    ) for k, v in value.items()}
                    result['input_ids'] = {k: t['input_ids'] for k, t in tokens.items()}
                    result['attention_mask'] = {k: t['attention_mask'] for k, t in tokens.items()}
                else:
                    logger.warning(f"Unknown caption data type for key '{key}': {type(value)}")
                    result['caption'] = value
            else:
                # Handle other columns
                result[key] = value
        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item by index.

        Args:
            idx (int): Index.

        Returns:
            Dict[str, Any]: Processed example data.
        """
        current_idx = idx
        for dataset in self.datasets:
            if current_idx < len(dataset):
                example = dataset[int(current_idx)]
                return self._process_example(example)
            else:
                current_idx -= len(dataset)
        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to batch data.

    Args:
        batch (List[Dict[str, Any]]): List of examples.

    Returns:
        Dict[str, Any]: Batched data.
    """
    batch_audio = [item['audio'] for item in batch if 'audio' in item]
    if all(isinstance(a, torch.Tensor) for a in batch_audio):
        batch_audio = torch.stack(batch_audio)
    batch_input_ids = torch.cat([item['input_ids'] for item in batch if 'input_ids' in item], dim=0)
    batch_attention_mask = torch.cat([item['attention_mask'] for item in batch if 'attention_mask' in item], dim=0)

    batched_data = {
        'audio': batch_audio,
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
    }

    # Include other keys
    keys = set().union(*(d.keys() for d in batch))
    for key in keys:
        if key not in batched_data:
            batched_data[key] = [d[key] for d in batch if key in d]

    return batched_data


# Example usage:

if __name__ == '__main__':
    # Define special tokens
    special_tokens = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}

    # Initialize dataset
    dataset = GeneralizedAudioCaptionDataset(
        dataset_names_or_paths=['your_dataset_name_or_path'],
        tokenizer_name='bert-base-uncased',
        split='train',
        columns=['audio', 'caption', 'other_columns'],
        max_length=128,
        padding='max_length',
        special_tokens=special_tokens,
        transform=None
    )

    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Iterate over data
    for batch in data_loader:
        audio = batch['audio']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # Training code goes here