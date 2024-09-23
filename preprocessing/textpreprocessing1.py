import logging
from typing import Any, Dict, List, Union, Optional
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader, Dataset, ChainDataset, ConcatDataset
import torch
import logging
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader, Dataset, ChainDataset, ConcatDataset
import os
import re
from typing import List, Dict
from transformers import PreTrainedTokenizer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TextFileDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing text files from a directory.
    Each file can have a variable number of lines and variable line lengths.
    """

    def __init__(
        self,
        directory: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Initializes the dataset by listing all .txt files in the provided directory.

        Args:
            directory (str): Path to the directory containing .txt files.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for processing text.
            max_length (Optional[int], optional): Maximum token length. Defaults to None.
        """
        super().__init__()
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token exists
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Gather all .txt file paths
        self.file_paths = self._get_txt_file_paths()
        # Filter out empty files
        self.file_paths = [fp for fp in self.file_paths if self._read_file(fp).strip()]
        if not self.file_paths:
            logging.warning(f"No non-empty .txt files found in directory: {directory}")

    def _get_txt_file_paths(self) -> List[str]:
        """
        Retrieves all .txt file paths from the specified directory.

        Returns:
            List[str]: List of .txt file paths.
        """
        try:
            files = [
                os.path.join(self.directory, fname)
                for fname in os.listdir(self.directory)
                if fname.lower().endswith(".txt")
            ]
            logging.info(f"Found {len(files)} .txt files in {self.directory}")
            return files
        except FileNotFoundError as e:
            logging.error(f"Directory not found: {self.directory}")
            raise e
        except Exception as e:
            logging.error(f"Error accessing directory {self.directory}: {e}")
            raise e

    def __len__(self) -> int:
        return len(self.file_paths)

    def _read_file(self, file_path: str) -> str:
        """
        Reads the content of a text file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Content of the file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return ""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves and tokenizes the text from the specified file index.

        Args:
            idx (int): Index of the file.

        Returns:
            Dict[str, Any]: Tokenized representation of the text.
        """
        file_path = self.file_paths[idx]
        text = self._read_file(file_path)

        if not text.strip():
            logging.warning(f"File {file_path} is empty.")
            # Optionally assign a default token or handle as needed
            text = self.tokenizer.pad_token or "[PAD]"

        # Encode the text with proper parameters
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,    # Let tokenizer add [CLS] and [SEP]
            truncation=True,           # Enable truncation
            max_length=self.max_length,
            padding='max_length' if self.max_length else False,  # Pad to max_length if specified
            return_tensors='pt',       # Return PyTorch tensors
            # clean_up_tokenization_spaces=False  # Handle future warning
        )

        # Remove batch dimension
        input_ids = encoded['input_ids'].squeeze()

        return {"input_ids": input_ids}


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Creates a DataLoader with appropriate settings and a data collator.

    Args:
        dataset (Dataset): The dataset to load.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for data collation.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        DataLoader: Configured DataLoader.
    """
    try:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,                 # Dynamic padding
            return_tensors='pt'           # Return PyTorch tensors
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=data_collator,
        )
        logging.info("DataLoader created successfully.")
        return dataloader
    except Exception as e:
        logging.error(f"Failed to create DataLoader: {e}")
        raise e
class GeneralizedDatasetLoader:
    """
    Generalized Dataset Loader that can load datasets with arbitrary splits and columns.

    Attributes:
        dataset_name (str): The name of the dataset to load.
        splits (Union[str, List[str]]): The splits to load. Can be a split name or a list of split names.
        columns (Union[str, List[str], Dict[str, str]]): The columns to keep or rename.
        **kwargs: Additional keyword arguments to pass to load_dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        splits: Union[str, List[str]] = "train",
        columns: Union[str, List[str], Dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        self.dataset_name = dataset_name
        self.splits = [splits] if isinstance(splits, str) else splits
        self.columns = columns
        self.kwargs = kwargs
        self.dataset: Optional[DatasetDict] = None

    def load(self) -> DatasetDict:
        """
        Load the dataset with specified splits and columns.

        Returns:
            DatasetDict: A dictionary of datasets for each split.
        """
        try:
            split_dict = {split: split for split in self.splits}
            self.dataset = load_dataset(
                self.dataset_name, split=split_dict, **self.kwargs
            )
            if self.columns:
                self.dataset = self._process_columns(self.dataset)
            return self.dataset
        except Exception as e:
            logger.error(
                f"Failed to load dataset {self.dataset_name} with splits {self.splits}: {e}"
            )
            raise

    def _process_columns(self, dataset: Dict[str, HFDataset]) -> Dict[str, HFDataset]:
        """
        Process columns by selecting or renaming them.

        Args:
            dataset (Dict[str, HFDataset]): The loaded dataset.

        Returns:
            Dict[str, HFDataset]: Dataset with processed columns.
        """
        try:
            for split in dataset:
                ds = dataset[split]
                if isinstance(self.columns, str):
                    ds = ds.select_columns([self.columns])
                elif isinstance(self.columns, list):
                    ds = ds.select_columns(self.columns)
                elif isinstance(self.columns, dict):
                    # Key: old column name, Value: new column name
                    old_columns = list(self.columns.keys())
                    ds = ds.select_columns(old_columns)
                    ds = ds.rename_columns(self.columns)
                else:
                    raise ValueError(f"Invalid type for columns: {type(self.columns)}")
                dataset[split] = ds
            return dataset
        except Exception as e:
            logger.error(f"Failed to process columns {self.columns}: {e}")
            raise


class RobustTokenizer:
    """
    Robust Tokenizer using transformers.AutoTokenizer API.

    Args:
        tokenizer_name (str): Name or path of the tokenizer.
        use_fast (bool): Whether to use the fast tokenizer.
        special_tokens (Dict[str, str]): Special tokens to add to the tokenizer.
        max_length (int): Maximum sequence length.
        padding (Union[bool, str]): Padding strategy.
        **kwargs: Additional kwargs for AutoTokenizer.from_pretrained.
    """

    def __init__(
        self,
        tokenizer_name: str,
        use_fast: bool = True,
        special_tokens: Optional[Dict[str, str]] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        **kwargs: Any
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.use_fast = use_fast
        self.special_tokens = special_tokens
        self.max_length = max_length
        self.padding = padding
        self.kwargs = kwargs
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self) -> None:
        """
        Initialize the tokenizer with special tokens.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, use_fast=self.use_fast, **self.kwargs
            )
            if self.special_tokens:
                self.tokenizer.add_special_tokens(self.special_tokens)
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer {self.tokenizer_name}: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text or list of texts.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to encode.
            **kwargs: Additional keyword arguments for tokenizer.encode / tokenizer.batch_encode_plus.

        Returns:
            Union[List[int], List[List[int]]]: Encoded token IDs.
        """
        try:
            kwargs.setdefault("max_length", self.max_length)
            kwargs.setdefault("padding", self.padding)
            kwargs.setdefault("truncation", True)

            if isinstance(texts, str):
                return self.tokenizer.encode(texts, **kwargs)
            elif isinstance(texts, list):
                return self.tokenizer(texts, **kwargs)["input_ids"]
            else:
                raise ValueError(f"Invalid type for texts: {type(texts)}")
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        **kwargs: Any
    ) -> Union[str, List[str]]:
        """
        Decode token IDs or list of token IDs.

        Args:
            token_ids (Union[List[int], List[List[int]]]): Token IDs to decode.
            **kwargs: Additional keyword arguments for tokenizer.decode / tokenizer.batch_decode.

        Returns:
            Union[str, List[str]]: Decoded text(s).
        """
        try:
            if isinstance(token_ids[0], int):
                return self.tokenizer.decode(token_ids, **kwargs)
            elif isinstance(token_ids[0], list):
                return self.tokenizer.batch_decode(token_ids, **kwargs)
            else:
                raise ValueError(f"Invalid type for token_ids: {type(token_ids)}")
        except Exception as e:
            logger.error(f"Failed to decode token IDs: {e}")
            raise


class HF2TorchDataset(Dataset):
    """
    Wrapper to convert a Hugging Face Dataset to a PyTorch Dataset.

    Args:
        hf_dataset (datasets.Dataset): The Hugging Face Dataset to wrap.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for processing text.
        text_column (str): Name of the text column.
        label_column (Optional[str]): Name of the label column, if any.
        max_length (Optional[int]): Max length for tokenization.
        padding (Union[bool, str]): Padding strategy for tokenization.
        **tokenizer_kwargs: Additional kwargs for tokenizer.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str,
        label_column: Optional[str] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        **tokenizer_kwargs: Any
    ) -> None:
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.padding = padding
        self.tokenizer_kwargs = tokenizer_kwargs

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.hf_dataset[idx]
        text = item[self.text_column]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
            **self.tokenizer_kwargs
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.label_column:
            encoding["labels"] = torch.tensor(item[self.label_column])
        return encoding


def create_dataloader(
    datasets_list: List[Dataset],
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Optional[Any] = None,
    **kwargs: Any
) -> DataLoader:
    """
    Create a DataLoader from a list of datasets.

    Args:
        datasets_list (List[Dataset]): List of PyTorch Datasets.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        collate_fn (Optional[Any]): Function to collate samples.
        **kwargs: Additional kwargs for DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    try:
        if len(datasets_list) == 1:
            dataset = datasets_list[0]
        else:
            dataset = ConcatDataset(datasets_list)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )
        return dataloader
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}")
        raise


# # Initialize the dataset loader
# dataset_loader = GeneralizedDatasetLoader(
#     dataset_name='imdb',  # Example dataset
#     splits=['train', 'test'],
#     columns={'text': 'review_text', 'label': 'sentiment'}
# )

# # Load the dataset
# datasets_dict = dataset_loader.load()

# # Initialize the tokenizer
# tokenizer = RobustTokenizer(
#     tokenizer_name='bert-base-uncased',
#     special_tokens={'bos_token': '[BOS]', 'eos_token': '[EOS]'},
#     max_length=128,
#     padding='max_length'
# )

# # Wrap datasets into PyTorch Datasets
# train_dataset = HF2TorchDataset(
#     hf_dataset=datasets_dict['train'],
#     tokenizer=tokenizer.tokenizer,
#     text_column='review_text',
#     label_column='sentiment',
#     max_length=128,
#     padding='max_length'
# )

# test_dataset = HF2TorchDataset(
#     hf_dataset=datasets_dict['test'],
#     tokenizer=tokenizer.tokenizer,
#     text_column='review_text',
#     label_column='sentiment',
#     max_length=128,
#     padding='max_length'
# )

# # Create data loaders
# train_loader = create_dataloader(
#     datasets_list=[train_dataset],
#     batch_size=32,
#     shuffle=True,
#     num_workers=4
# )

# test_loader = create_dataloader(
#     datasets_list=[test_dataset],
#     batch_size=32,
#     shuffle=False,
#     num_workers=4
# )





class DataManager:
    """A class for managing datasets and tokenization for NLP tasks.
    
    This class provides methods for loading datasets, setting up tokenizers,
    tokenizing datasets, combining datasets, and creating data loaders.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DataManager with a tokenizer.
        
        Args:
            tokenizer_name_or_path: The name or path of the tokenizer.
            tokenizer_kwargs: Additional keyword arguments for the tokenizer.
            kwargs: Additional keyword arguments.
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.datasets: Dict[str, HFDataset] = {}

        self.setup_tokenizer(**kwargs)

    def setup_tokenizer(self, **kwargs: Any) -> None:
        """Set up the tokenizer with the specified options.
        
        Args:
            kwargs: Additional keyword arguments for tokenizer configuration.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path, **self.tokenizer_kwargs
            )

            # Update tokenizer with additional configurations
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
            if "bos_token" in kwargs:
                self.tokenizer.bos_token = kwargs["bos_token"]
            if "eos_token" in kwargs:
                self.tokenizer.eos_token = kwargs["eos_token"]
            if "additional_special_tokens" in kwargs:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": kwargs["additional_special_tokens"]}
                )
        except Exception as e:
            logging.error(f"Error setting up tokenizer: {e}")
            raise e

    def load_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        split: Optional[Union[str, List[str]]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Load a dataset and store it internally.
        
        Args:
            path: Path or name of the dataset.
            name: Name of the dataset configuration.
            data_dir: Directory containing the dataset files.
            split: Which split(s) to load.
            cache_dir: Where to cache the dataset.
            features: Features type of the dataset.
            kwargs: Additional keyword arguments.
        """
        try:
            dataset = load_dataset(
                path,
                name=name,
                data_dir=data_dir,
                split=split,
                cache_dir=cache_dir,
                features=features,
                **kwargs,
            )
            if isinstance(split, list):
                for s, d in zip(split, dataset):
                    self.datasets[s] = d
            else:
                split_name = split if split else "all"
                self.datasets[split_name] = dataset
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise e
    # Modify the part where the error occurs in the tokenize_dataset method

    def tokenize_dataset(
        self,
        columns: Union[str, List[str], Dict[str, List[str]]],
        batched: bool = True,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = True,
        **kwargs: Any,
) ->     None:
        """Tokenize the loaded datasets.
        
        Args:
            columns: The column(s) to tokenize.
            batched: Whether to tokenize in batches.
            batch_size: The batch size for tokenization.
            max_length: The maximum sequence length.
            padding: Padding strategy.
            truncation: Truncation strategy.
            kwargs: Additional keyword arguments for tokenization.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been set up.")
        if not self.datasets:
            raise ValueError("No datasets have been loaded.")
    
        try:
            
    
            def tokenize_function(examples):
               if isinstance(columns, str):
                   texts = examples[columns]
               elif isinstance(columns, list):
                   texts = []
                   for items in zip(*[examples[col] for col in columns]):
                       # Check if any element in the items is a list
                       processed_items = []
                       for item in items:
                           if isinstance(item, list):
                               # If it's a list, join the elements as a string
                               processed_items.append(" ".join(map(str, item)))
                           else:
                               processed_items.append(str(item))
                       # Join the processed items into a single string
                       texts.append(" ".join(processed_items))
               elif isinstance(columns, dict):
                   texts = []
                   for idx in range(len(next(iter(examples.values())))):
                       text_parts = []
                       for key, cols in columns.items():
                           part = " ".join([str(examples[col][idx]) if not isinstance(examples[col][idx], list) else " ".join(map(str, examples[col][idx])) for col in cols])
                           text_parts.append(part)
                       texts.append(" ".join(text_parts))
               else:
                   raise ValueError(f"Unsupported column type: {type(columns)}")
           
               return self.tokenizer(
                   texts,
                   max_length=max_length,
                   padding=padding,
                   truncation=truncation,
                   **kwargs,
               )

    
            for split_name, dataset in self.datasets.items():
                self.datasets[split_name] = dataset.map(
                    tokenize_function,
                    batched=batched,
                    batch_size=batch_size,
                )
                # Set the format for PyTorch
                self.datasets[split_name].set_format(
                    type="torch", columns=["input_ids", "attention_mask"]
                )
        except Exception as e:
            logging.error(f"Error tokenizing dataset: {e}")
            raise e

    # def tokenize_dataset(
    #     self,
    #     columns: Union[str, List[str], Dict[str, List[str]]],
    #     batched: bool = True,
    #     batch_size: Optional[int] = None,
    #     max_length: Optional[int] = None,
    #     padding: Union[bool, str] = False,
    #     truncation: Union[bool, str] = True,
    #     **kwargs: Any,
    # ) -> None:
    #     """Tokenize the loaded datasets.
        
    #     Args:
    #         columns: The column(s) to tokenize.
    #         batched: Whether to tokenize in batches.
    #         batch_size: The batch size for tokenization.
    #         max_length: The maximum sequence length.
    #         padding: Padding strategy.
    #         truncation: Truncation strategy.
    #         kwargs: Additional keyword arguments for tokenization.
    #     """
    #     if not self.tokenizer:
    #         raise ValueError("Tokenizer has not been set up.")
    #     if not self.datasets:
    #         raise ValueError("No datasets have been loaded.")

    #     try:

    #         def tokenize_function(examples):
    #             if isinstance(columns, str):
    #                 texts = examples[columns]
    #             elif isinstance(columns, list):
    #                 texts = [
    #                     " ".join(items)
    #                     for items in zip(*[examples[col] for col in columns])
    #                 ]
    #             elif isinstance(columns, dict):
    #                 texts = []
    #                 for idx in range(len(next(iter(examples.values())))):
    #                     text_parts = []
    #                     for key, cols in columns.items():
    #                         part = " ".join([examples[col][idx] for col in cols])
    #                         text_parts.append(part)
    #                     texts.append(" ".join(text_parts))
    #             else:
    #                 raise ValueError(f"Unsupported column type: {type(columns)}")

    #             return self.tokenizer(
    #                 texts,
    #                 max_length=max_length,
    #                 padding=padding,
    #                 truncation=truncation,
    #                 **kwargs,
    #             )

    #         for split_name, dataset in self.datasets.items():
    #             self.datasets[split_name] = dataset.map(
    #                 tokenize_function,
    #                 batched=batched,
    #                 batch_size=batch_size,
    #             )
    #             # Set the format for PyTorch
    #             self.datasets[split_name].set_format(
    #                 type="torch", columns=["input_ids", "attention_mask"]
    #             )
    #     except Exception as e:
    #         logging.error(f"Error tokenizing dataset: {e}")
    #         raise e

    def combine_datasets(
        self,
        datasets: List[Dataset],
        method: str = "concat",
        **kwargs: Any,
    ) -> Dataset:
        """Combine multiple datasets into one.
        
        Args:
            datasets: List of datasets to combine.
            method: Method of combining ('concat' or 'chain').
            kwargs: Additional keyword arguments.
            
        Returns:
            The combined dataset.
        """
        try:
            if method == "concat":
                combined_dataset = ConcatDataset(datasets)
            elif method == "chain":
                combined_dataset = ChainDataset(datasets)
            else:
                raise ValueError(f"Unknown combination method: {method}")
            return combined_dataset
        except Exception as e:
            logging.error(f"Error combining datasets: {e}")
            raise e

    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
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
        """Create a DataLoader for the given dataset.
        
        Args:
            dataset: The dataset to load.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
            num_workers: Number of subprocesses to use for data loading.
            collate_fn: Merges a list of samples to form a mini-batch.
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory.
            drop_last: Whether to drop the last incomplete batch.
            timeout: Timeout for collecting a batch from workers.
            worker_init_fn: Function to be called on each worker subprocess.
            prefetch_factor: Number of batches loaded in advance.
            persistent_workers: Whether to keep workers alive after dataset has been consumed once.
            kwargs: Additional keyword arguments.
            
        Returns:
            A DataLoader instance.
        """
        try:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
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
            return dataloader
        except Exception as e:
            logging.error(f"Error creating DataLoader: {e}")
            raise e

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = True,
        **kwargs: Any,
    ) -> Union[List[int], List[List[int]]]:
        """Encode a single text or a batch of texts.
        
        Args:
            text: Text(s) to encode.
            max_length: Maximum length of the encoded text.
            padding: Padding strategy.
            truncation: Truncation strategy.
            kwargs: Additional keyword arguments.
            
        Returns:
            Encoded text(s).
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been set up.")
        if isinstance(text, str):
            return self.tokenizer.encode(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )
        elif isinstance(text, list):
            return self.tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )["input_ids"]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Decode a single sequence or a batch of sequences.
        
        Args:
            token_ids: Token IDs to decode.
            skip_special_tokens: Whether to skip special tokens.
            kwargs: Additional keyword arguments.
            
        Returns:
            Decoded text(s).
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been set up.")
        if isinstance(token_ids[0], int):
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )
        elif isinstance(token_ids[0], list):
            return self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported token_ids type: {type(token_ids)}")




def read_txt_files_with_labels(directory: str) -> List[Dict[str, str]]:
    data = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    if not lines:
                        continue
                    label = lines[0].strip()  # Adjust based on how labels are stored
                    content = '\n'.join(line.strip() for line in lines[1:] if line.strip())
                    data.append({'text': content, 'label': label})
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
    return data

def split_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> List[str]:
    words = text.strip().split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        tokenized_word = tokenizer.encode(word, add_special_tokens=False)
        word_length = len(tokenized_word)
        if current_length + word_length > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def tokenize_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> Dict[str, List[List[int]]]:
    all_input_ids = []
    all_attention_masks = []
    for text in texts:
        if not text:
            continue
        chunks = split_text(text, tokenizer, max_length - 2)
        for chunk in chunks:
            tokens = tokenizer.encode(
                chunk,
                add_special_tokens=False,
                max_length=max_length - 2,
                truncation=True
            )
            tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
            attention_mask = [1] * len(tokens)
            if len(tokens) < max_length:
                padding_length = max_length - len(tokens)
                tokens += [tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            else:
                tokens = tokens[:max_length]
                attention_mask = attention_mask[:max_length]
            all_input_ids.append(tokens)
            all_attention_masks.append(attention_mask)
    return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks}


# # Initialize the DataManager with a tokenizer
# data_manager = DataManager(
#     tokenizer_name_or_path='bert-base-uncased',
#     tokenizer_kwargs={'use_fast': True},
#     bos_token='[BOS]',
#     eos_token='[EOS]',
#     additional_special_tokens=['[SPECIAL1]', '[SPECIAL2]']
# )

# # Load datasets with various configurations
# data_manager.load_dataset(
#     path='your_dataset_path',
#     split=['train', 'validation', 'test'],
#     cache_dir='./cache'
# )

# # Tokenize the datasets
# data_manager.tokenize_dataset(
#     columns={'text': ['column1', 'column2']},
#     max_length=128,
#     padding='max_length',
#     truncation=True
# )

# # Combine datasets if needed
# combined_dataset = data_manager.combine_datasets(
#     datasets=[data_manager.datasets['train'], data_manager.datasets['validation']],
#     method='concat'
# )

# # Create a DataLoader
# train_loader = data_manager.get_dataloader(
#     dataset=combined_dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )

# # Encode and decode examples
# encoded_input = data_manager.encode("Example text to encode.")
# decoded_output = data_manager.decode(encoded_input)



def read_txt_files(directory: str) -> List[str]:
    """
    Reads all .txt files from the given directory and returns a list of their contents.

    Args:
        directory (str): The path to the directory containing .txt files.

    Returns:
        List[str]: A list of strings, each string is the content of a .txt file.
    """
    contents = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Clean up the content by stripping whitespace and handling empty lines
                    content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
                    contents.append(content)
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
    return contents


def split_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> List[str]:
    """
    Splits a text into chunks that fit within the max_length when tokenized.

    Args:
        text (str): The text to split.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum length of the tokenized output.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.strip().split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        # Encode the word to get its token length
        tokenized_word = tokenizer.encode(word, add_special_tokens=False)
        word_length = len(tokenized_word)
        # Check if adding the word exceeds the max_length
        if current_length + word_length > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def tokenize_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int
) -> Dict[str, List[List[int]]]:
    """
    Tokenizes a list of texts using the given tokenizer, ensuring zero data loss by splitting long texts.
    Adds BOS and EOS tokens manually.

    Args:
        texts (List[str]): The texts to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum length of the tokenized output.

    Returns:
        Dict[str, List[List[int]]]: A dictionary containing 'input_ids' and 'attention_mask'.
    """
    all_input_ids = []
    all_attention_masks = []
    for text in texts:
        # Skip empty texts
        if not text:
            continue
        chunks = split_text(text, tokenizer, max_length - 2)  # Reserve space for BOS and EOS tokens
        for chunk in chunks:
            # Tokenize the chunk without adding any special tokens
            tokens = tokenizer.encode(
                chunk,
                add_special_tokens=False,
                max_length=max_length - 2,
                truncation=True
            )
            # Manually add BOS and EOS tokens
            tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
            # Create attention mask
            attention_mask = [1] * len(tokens)
            # Pad if necessary
            if len(tokens) < max_length:
                padding_length = max_length - len(tokens)
                tokens += [tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            else:
                tokens = tokens[:max_length]
                attention_mask = attention_mask[:max_length]
            all_input_ids.append(tokens)
            all_attention_masks.append(attention_mask)
    return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks}

