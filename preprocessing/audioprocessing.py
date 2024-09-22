import os
import logging
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union

import requests
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, ChainDataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralizedAudioCaptionDataset(Dataset):
    """Generalized Dataset for loading audio and caption data.

    This dataset handles any splits and number of columns,
    with columns of type str, List[str], or Dict[str, List[str]].
    Audio paths can be absolute paths or URLs.

    Args:
        data_list (List[Dict[str, Any]]): List containing dataset entries.
        transform (Optional[Callable]): Optional transform to apply on a sample.
        *args: Variable length argument list for `librosa.load`.
        **kwargs: Arbitrary keyword arguments for `librosa.load`.
    """

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any
    ):
        self.data_list = data_list
        self.transform = transform
        self.args = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            sample = self.data_list[idx]
            data: Dict[str, Any] = {}

            # Process each column
            for key, value in sample.items():
                if key == 'audio':
                    # Load and process audio
                    audio = self.load_audio(value)
                    data['audio'] = audio
                else:
                    # Handle other columns (str, list, dict)
                    data[key] = value

            if self.transform:
                data = self.transform(data)

            return data

        except Exception as e:
            logger.error(f"Error loading data at index {idx}: {e}")
            raise

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio from a given path or URL.

        Args:
            audio_path (str): Path or URL to the audio file.

        Returns:
            torch.Tensor: Loaded audio tensor.
        """
        try:
            if audio_path.startswith(('http://', 'https://')):
                # Download the file
                response = requests.get(audio_path)
                if response.status_code != 200:
                    raise IOError(
                        f"Failed to download audio from '{audio_path}'. "
                        f"Status code: {response.status_code}"
                    )
                audio_file = BytesIO(response.content)
            else:
                # Load from local file system
                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                audio_file = audio_path

            # Load audio using librosa (supports various formats)
            audio_data, _ = librosa.load(audio_file, *self.args, **self.kwargs)

            # Convert to torch tensor
            audio_tensor = torch.tensor(audio_data)
            return audio_tensor

        except Exception as e:
            logger.error(f"Error loading audio '{audio_path}': {e}")
            raise


class RobustTokenizer:
    """Robust Tokenizer using Hugging Face AutoTokenizer.

    Implements special tokens, BOS, EOS, padding, max_length,
    batch encoding and decoding.

    Args:
        tokenizer_name_or_path (str): Name or path to the tokenizer.
        bos_token (Optional[str]): Beginning of sentence token.
        eos_token (Optional[str]): End of sentence token.
        padding (Union[bool, str]): Padding strategy ('max_length', 'longest', etc.).
        max_length (Optional[int]): Maximum sequence length.
        *args: Variable length argument list for tokenizer.
        **kwargs: Arbitrary keyword arguments for tokenizer.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        *args: Any,
        **kwargs: Any
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, *args, **kwargs
        )
        # Set special tokens if provided
        if bos_token:
            self.tokenizer.bos_token = bos_token
        if eos_token:
            self.tokenizer.eos_token = eos_token

        self.padding = padding
        self.max_length = max_length

    def encode(
        self,
        text: Union[str, List[str]],
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Encode text into tokens.

        Args:
            text (Union[str, List[str]]): Text or list of texts to encode.
            *args: Variable length argument list for tokenizer.
            **kwargs: Arbitrary keyword arguments for tokenizer.

        Returns:
            Dict[str, torch.Tensor]: Encoded inputs.
        """
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                *args,
                **kwargs
            )
            return encoding
        except Exception as e:
            logger.error(f"Error encoding text '{text}': {e}")
            raise

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        *args: Any,
        **kwargs: Any
    ) -> Union[str, List[str]]:
        """Decode tokens back to text.

        Args:
            token_ids (Union[List[int], torch.Tensor]): Token IDs to decode.
            *args: Variable length argument list for tokenizer.
            **kwargs: Arbitrary keyword arguments for tokenizer.

        Returns:
            Union[str, List[str]]: Decoded text or list of texts.
        """
        try:
            text = self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=True,
                *args,
                **kwargs
            )
            return text
        except Exception as e:
            logger.error(f"Error decoding token IDs '{token_ids}': {e}")
            raise




def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[Any] = None,
    batch_sampler: Optional[Any] = None,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Optional[Callable] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    *args: Any,
    **kwargs: Any
) -> DataLoader:
    """Create a DataLoader with given arguments.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch_size (int, optional): Number of samples per batch.
        shuffle (bool, optional): Shuffle data at every epoch.
        sampler (Optional[Any], optional): Strategy to draw samples.
        batch_sampler (Optional[Any], optional): Returns a batch of indices at a time.
        num_workers (int, optional): Subprocesses for data loading.
        collate_fn (Optional[Callable], optional): Merges samples to form a mini-batch.
        pin_memory (bool, optional): Copy Tensors into CUDA pinned memory.
        drop_last (bool, optional): Drop last incomplete batch.
        timeout (float, optional): Timeout for collecting a batch.
        worker_init_fn (Optional[Callable], optional): Called on each worker subprocess.
        prefetch_factor (int, optional): Batches loaded in advance per worker.
        persistent_workers (bool, optional): Do not shutdown workers between epochs.
        *args: Variable length argument list for DataLoader.
        **kwargs: Arbitrary keyword arguments for DataLoader.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    try:
        dataloader = DataLoader(
            dataset=dataset,
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
        return dataloader
    except Exception as e:
        logger.error(f"Error creating DataLoader: {e}")
        raise


def combine_datasets(
    datasets: List[Dataset],
    method: str = 'concat',
    *args: Any,
    **kwargs: Any
) -> Dataset:
    """Combine multiple datasets.

    Args:
        datasets (List[Dataset]): Datasets to combine.
        method (str, optional): 'concat' to concatenate or 'chain' to chain datasets.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Dataset: Combined dataset.
    """
    try:
        if method == 'concat':
            combined_dataset = ConcatDataset(datasets)
        elif method == 'chain':
            combined_dataset = ChainDataset(datasets)
        else:
            raise ValueError(
                f"Unknown combination method '{method}'. Use 'concat' or 'chain'."
            )
        return combined_dataset
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")
        raise



# Sample data list with mixed columns
data_list = [
    {
        'audio': 'path_or_url_to_audio_1.wav',
        'caption': 'First audio caption.',
        'metadata': {'speaker': 'Alice', 'emotion': 'happy'}
    },
    {
        'audio': 'path_or_url_to_audio_2.mp3',
        'caption': ['Second audio caption part one.', 'Second audio caption part two.'],
        'metadata': {'speaker': 'Bob', 'emotion': 'sad'}
    }
]

# Create dataset instance
dataset = GeneralizedAudioCaptionDataset(data_list)

# Initialize tokenizer
tokenizer = RobustTokenizer(
    tokenizer_name_or_path='bert-base-uncased',
    bos_token='[BOS]',
    eos_token='[EOS]',
    padding='max_length',
    max_length=128
)

# Create DataLoader
dataloader = create_dataloader(
    dataset=dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate over DataLoader
for batch in dataloader:
    audio_batch = batch['audio']
    captions = batch['caption']

    # Tokenize captions
    encoded_captions = tokenizer.encode(captions)
    input_ids = encoded_captions['input_ids']
    attention_mask = encoded_captions['attention_mask']

    # Proceed with model training or inference








import os
import requests
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Union

import librosa
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset
from transformers import AutoTokenizer


class GeneralAudioCaptionDataset(Dataset):
    """
    A general-purpose dataset loader for audio and caption data.
    Supports loading from local paths or URLs, handling various
    column types, and integrates with Hugging Face datasets.
    """

    def __init__(
        self,
        dataset_name: str,
        split: Union[str, List[str]] = "train",
        audio_column: str = "audio",
        caption_column: str = "caption",
        additional_columns: Optional[List[str]] = None,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        padding: Union[str, bool] = "max_length",
        truncation: bool = True,
        special_tokens: Optional[Dict[str, str]] = None,
        num_proc: int = 1,
        use_auth_token: Optional[Union[bool, str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initializes the dataset by loading it from the Hugging Face datasets.

        Args:
            dataset_name: The name of the dataset to load.
            split: The dataset split(s) to use.
            audio_column: The name of the audio column.
            caption_column: The name of the caption column.
            additional_columns: Any additional columns to load.
            tokenizer_name: The name of the tokenizer to use.
            max_length: The maximum sequence length for tokenization.
            padding: Padding strategy for tokenization.
            truncation: Whether to truncate sequences.
            special_tokens: Special tokens to add to the tokenizer.
            num_proc: Number of processes to use for data processing.
            use_auth_token: Authentication token for private datasets.
            cache_dir: Directory to cache the dataset.
        """
        self.dataset: Union[HFDataset, DatasetDict] = load_dataset(
            dataset_name,
            split=split,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
        )

        self.audio_column = audio_column
        self.caption_column = caption_column
        self.additional_columns = additional_columns or []

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if special_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(special_tokens.values())}
            )

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        self.num_proc = num_proc

        # Preprocess the dataset to prepare audio and caption data
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """Prepares the dataset by loading audio and tokenizing captions."""

        def load_audio(example: Dict[str, Any]) -> Dict[str, Any]:
            audio_path = example[self.audio_column]
            try:
                if audio_path.startswith(("http://", "https://")):
                    # Handle URL audio
                    with tempfile.NamedTemporaryFile(
                        delete=True, suffix=os.path.splitext(audio_path)[1]
                    ) as tmp_file:
                        response = requests.get(audio_path, stream=True)
                        if response.status_code == 200:
                            for chunk in response.iter_content(chunk_size=8192):
                                tmp_file.write(chunk)
                            tmp_file.flush()
                            audio, sample_rate = librosa.load(
                                tmp_file.name, sr=None
                            )
                            example["audio_array"] = audio
                            example["sample_rate"] = sample_rate
                        else:
                            warnings.warn(
                                f"Failed to download audio from {audio_path}, "
                                f"status code {response.status_code}"
                            )
                            example["audio_array"] = None
                            example["sample_rate"] = None
                else:
                    # Local file
                    if os.path.exists(audio_path):
                        audio, sample_rate = librosa.load(audio_path, sr=None)
                        example["audio_array"] = audio
                        example["sample_rate"] = sample_rate
                    else:
                        warnings.warn(
                            f"Audio file not found at {audio_path}"
                        )
                        example["audio_array"] = None
                        example["sample_rate"] = None
            except Exception as e:
                warnings.warn(f"Error loading audio at {audio_path}: {e}")
                example["audio_array"] = None
                example["sample_rate"] = None
            return example

        def tokenize_caption(example: Dict[str, Any]) -> Dict[str, Any]:
            caption = example[self.caption_column]
            try:
                if isinstance(caption, str):
                    pass
                elif isinstance(caption, list):
                    caption = " ".join(caption)
                elif isinstance(caption, dict):
                    # Flatten the dict
                    caption = " ".join(
                        [" ".join(v) for v in caption.values()]
                    )
                else:
                    warnings.warn(f"Unrecognized caption format: {caption}")
                    caption = ""
                tokens = self.tokenizer.encode_plus(
                    caption,
                    max_length=self.max_length,
                    padding=self.padding,
                    truncation=self.truncation,
                    return_tensors="pt",
                )
                example["caption_input_ids"] = tokens["input_ids"].squeeze(0)
                example["caption_attention_mask"] = tokens[
                    "attention_mask"
                ].squeeze(0)
            except Exception as e:
                warnings.warn(f"Error tokenizing caption: {e}")
                example["caption_input_ids"] = torch.tensor([])
                example["caption_attention_mask"] = torch.tensor([])
            return example

        # Map functions over the dataset with error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dataset = self.dataset.map(
                load_audio,
                remove_columns=[self.audio_column],
                num_proc=self.num_proc,
            )

            self.dataset = self.dataset.filter(
                lambda ex: ex["audio_array"] is not None
                and ex["sample_rate"] is not None
            )

            self.dataset = self.dataset.map(
                tokenize_caption,
                remove_columns=[self.caption_column],
                num_proc=self.num_proc,
            )

            self.dataset = self.dataset.filter(
                lambda ex: len(ex["caption_input_ids"]) > 0
            )

        # Keep necessary columns
        columns_to_keep = [
            "audio_array",
            "sample_rate",
            "caption_input_ids",
            "caption_attention_mask",
        ] + self.additional_columns
        self.dataset.set_format(type="torch", columns=columns_to_keep)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if item["audio_array"] is None:
            raise ValueError(f"Missing audio data at index {idx}.")
        if len(item["caption_input_ids"]) == 0:
            raise ValueError(f"Empty caption at index {idx}.")
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable length audio and captions.

    Args:
        batch: A list of dataset items.

    Returns:
        A dictionary with batched and padded data.
    """
    # Collate audio arrays (variable length)
    audio_arrays = [item["audio_array"] for item in batch]
    sample_rates = [item["sample_rate"] for item in batch]

    # Collate captions with padding
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["caption_input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        [item["caption_attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    # Collect additional columns
    additional_data = {}
    keys_to_exclude = {
        "audio_array",
        "sample_rate",
        "caption_input_ids",
        "caption_attention_mask",
    }
    for key in batch[0]:
        if key not in keys_to_exclude:
            additional_data[key] = [item[key] for item in batch]

    return {
        "audio_array": audio_arrays,
        "sample_rate": sample_rates,
        "caption_input_ids": input_ids,
        "caption_attention_mask": attention_masks,
        **additional_data,
    }


# Example usage
if __name__ == "__main__":
    # Create datasets
    dataset1 = GeneralAudioCaptionDataset(
        dataset_name="your_dataset1",
        split="train",
        audio_column="audio_path",
        caption_column="captions",
        tokenizer_name="bert-base-uncased",
        special_tokens={"bos_token": "<s>", "eos_token": "</s>"},
        num_proc=4,
    )

    dataset2 = GeneralAudioCaptionDataset(
        dataset_name="your_dataset2",
        split="train",
        audio_column="audio_path",
        caption_column="captions",
        tokenizer_name="bert-base-uncased",
        num_proc=4,
    )

    # Combine datasets
    combined_dataset = ConcatDataset([dataset1, dataset2])

    # Initialize DataLoader
    data_loader = DataLoader(
        combined_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        prefetch_factor=2,
        persistent_workers=False,
    )

    # Iterate over the DataLoader
    for batch in data_loader:
        # Process your batch here
        pass