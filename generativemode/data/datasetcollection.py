
import logging
from typing import List, Optional

from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset as TorchDataset


class HFDatasetWrapper(TorchDataset):
    """
    A wrapper class to make Hugging Face datasets compatible with PyTorch Dataset.
    """

    def __init__(self, dataset: HFDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]


def load_and_concat_datasets(dataset_names: List[str]) -> Optional[TorchDataset]:
    """
    Load and concatenate datasets given a list of dataset names.

    Args:
        dataset_names (List[str]): List of dataset names to load and concatenate.

    Returns:
        Optional[TorchDataset]: Returns concatenated dataset or None if loading fails.
    """
    logging.basicConfig(level=logging.INFO)
    datasets = []
    common_columns = None

    for name in dataset_names:
        dataset = None
        splits = ["train", "validation", "test"]
        for split in splits:
            try:
                logging.info(f"Trying to load dataset '{name}' split '{split}'.")
                dataset = load_dataset(name, split=split)
                logging.info(f"Successfully loaded dataset '{name}' split '{split}'.")
                break  # Exit the split loop if successful
            except Exception as e:
                logging.warning(
                    f"Failed to load split '{split}' for dataset '{name}': {e}"
                )
                continue  # Try next split

        if dataset is None:
            logging.error(
                f"All splits failed for dataset '{name}'. Skipping this dataset."
            )
            continue  # Skip to the next dataset

        dataset_columns = set(dataset.column_names)
        if common_columns is None:
            common_columns = dataset_columns
        else:
            common_columns = common_columns.intersection(dataset_columns)

        datasets.append(dataset)

    if not datasets:
        logging.error("No datasets were successfully loaded.")
        return None

    if not common_columns:
        logging.error("No common columns found across datasets.")
        return None

    # Keep only common columns in all datasets
    datasets = [
        dataset.remove_columns(set(dataset.column_names) - common_columns)
        for dataset in datasets
    ]

    # Convert datasets to PyTorch format
    torch_datasets = []
    for dataset in datasets:
        dataset.set_format(type="torch", columns=list(common_columns))
        torch_dataset = HFDatasetWrapper(dataset)
        torch_datasets.append(torch_dataset)

    # Use ConcatDataset to concatenate datasets
    concatenated_dataset = ConcatDataset(torch_datasets)

    logging.info("Datasets successfully concatenated.")

    return concatenated_dataset



from typing import List, Any, Set
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data.dataset import Dataset as TorchDataset, ConcatDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFDatasetWrapper(TorchDataset):
    """
    A wrapper to adapt a HuggingFace Dataset to PyTorch's Dataset interface,
    ensuring only specified columns are returned.
    """
    def __init__(self, hf_dataset: HFDataset, columns: List[str]) -> None:
        self.hf_dataset = hf_dataset
        self.columns = columns

    def __getitem__(self, idx: int) -> Any:
        item = self.hf_dataset[int(idx)]
        return {col: item[col] for col in self.columns}

    def __len__(self) -> int:
        return len(self.hf_dataset)


def load_and_concat_datasets(dataset_names: List[str]) -> TorchDataset:
    """
    Loads multiple datasets by name, selecting common columns, and concatenates them.

    Parameters:
        dataset_names (List[str]): List of dataset names to load.

    Returns:
        TorchDataset: Concatenated PyTorch Dataset containing data from all datasets.
    """
    datasets_list = []
    datasets_columns: List[Set[str]] = []
    datasets_splits = []

    for name in dataset_names:
        try:
            # Load the dataset by name
            dataset = load_dataset(name)
            logger.info(f"Loaded dataset '{name}' successfully.")

            # Select the 'train' split if available, else use the first available split
            if 'train' in dataset:
                split_dataset = dataset['train']
            else:
                split_name = next(iter(dataset.keys()))
                split_dataset = dataset[split_name]
                logger.warning(
                    f"Dataset '{name}' does not have a 'train' split. "
                    f"Using split '{split_name}' instead."
                )

            datasets_splits.append(split_dataset)
            datasets_columns.append(set(split_dataset.column_names))

        except Exception as e:
            logger.error(f"Error loading dataset '{name}': {e}")
            continue

    if not datasets_splits:
        raise ValueError("No datasets were loaded successfully.")

    # Find common columns among all datasets
    common_columns = set.intersection(*datasets_columns)
    if not common_columns:
        raise ValueError("No common columns found among the datasets.")

    common_columns = list(common_columns)
    logger.info(f"Common columns among datasets: {common_columns}")

    for split_dataset in datasets_splits:
        # Wrap the dataset and select common columns
        wrapped_dataset = HFDatasetWrapper(split_dataset, common_columns)
        datasets_list.append(wrapped_dataset)

    concatenated_dataset = ConcatDataset(datasets_list)
    logger.info("Datasets concatenated successfully.")
    return concatenated_dataset


from typing import Dict, Any, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    A class to generate prompts from templates and tokenize them using advanced settings.
    """
    def __init__(
        self,
        tokenizer_name: str,
        template: str,
        max_length: int = 512,
        padding: Union[bool, str] = True,
    ) -> None:
        """
        Initializes the PromptTemplate with a tokenizer and template.

        Args:
            tokenizer_name (str): Name or path of the pretrained tokenizer.
            template (str): The template string with placeholders for formatting.
            max_length (int, optional): Maximum sequence length. Defaults to 512.
            padding (Union[bool, str], optional): Padding strategy. Defaults to True (pad to max length).
        """
        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
            self.template: str = template
            self.max_length: int = max_length
            self.padding: Union[bool, str] = padding
            logger.info(f"Initialized PromptTemplate with tokenizer '{tokenizer_name}'.")
        except Exception as e:
            logger.error(f"Error initializing tokenizer '{tokenizer_name}': {e}")
            raise ValueError(f"Error initializing tokenizer '{tokenizer_name}': {e}")

    def __call__(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Generates a prompt by formatting the template with provided arguments and tokenizes it.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing tokenized tensors:
                - 'input_ids': Tensor of token ids.
                - 'labels': Tensor of labels for language modeling.
                - 'masked_ids': Tensor indicating special tokens.
                - 'attention_mask': Tensor of attention masks.
        """
        try:
            # Render the template with provided arguments
            prompt: str = self.template.format(**kwargs)
            logger.debug(f"Generated prompt: {prompt}")

            # Tokenization with advanced settings
            encoded: Dict[str, torch.Tensor] = self.tokenizer.encode_plus(
                prompt,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length' if self.padding else False,
                truncation=True,
                return_tensors='pt',
                return_special_tokens_mask=True,
                return_attention_mask=True,
            )

            # Extract required tensors
            input_ids: torch.Tensor = encoded['input_ids'].squeeze(0)
            attention_mask: torch.Tensor = encoded['attention_mask'].squeeze(0)
            masked_ids: torch.Tensor = encoded['special_tokens_mask'].squeeze(0)

            # Create labels (for language modeling, labels can be same as input_ids)
            labels: torch.Tensor = input_ids.clone()

            logger.debug("Tokenization successful.")
            return {
                'input_ids': input_ids,
                'labels': labels,
                'masked_ids': masked_ids,
                'attention_mask': attention_mask,
            }

        except KeyError as e:
            logger.error(f"Missing key for template rendering: {e}")
            raise ValueError(f"Missing key for template rendering: {e}")
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise ValueError(f"Error during tokenization: {e}")
        



from typing import List, Dict, Any, Union
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class PromptTemplate:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
        padding: Union[bool, str] = 'max_length',
        truncation: Union[bool, str] = True,
        use_special_tokens: bool = True,
        mlm_probability: float = 0.15,
        **kwargs: Any
    ) -> None:
        """
        Initializes the PromptTemplate with the specified tokenizer.

        Args:
            model_name_or_path (str): Pretrained tokenizer name or path.
            max_length (int, optional): Maximum length of the tokenized sequences.
                Defaults to 512.
            padding (Union[bool, str], optional): Padding strategy.
                Defaults to 'max_length'.
            truncation (Union[bool, str], optional): Truncation strategy.
                Defaults to True.
            use_special_tokens (bool, optional): Whether to include special tokens.
                Defaults to True.
            mlm_probability (float, optional): Probability of masking tokens for MLM.
                Defaults to 0.15.
            **kwargs: Additional keyword arguments to pass to the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.use_special_tokens = use_special_tokens
        self.mlm_probability = mlm_probability

    def __call__(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes and encodes the input text, returning input_ids, labels,
        and masked_ids.

        Args:
            text (Union[str, List[str]]): The input text or a list of input texts
                to tokenize.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels',
                and 'masked_ids'.
        """
        try:
            if isinstance(text, str):
                texts = [text]
            elif isinstance(text, list):
                texts = text
            else:
                raise ValueError("Input text must be a string or a list of strings.")

            # Tokenize the texts
            encoding: BatchEncoding = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=self.use_special_tokens,
                return_attention_mask=True,
                return_token_type_ids=True,
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # Copy input_ids to create labels
            labels = input_ids.clone()

            # Create masked_input_ids starting from input_ids
            masked_input_ids = input_ids.clone()

            # Create a mask for MLM
            # Mask mlm_probability% of the tokens
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                ) for val in labels.tolist()
            ]
            probability_matrix.masked_fill_(
                torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
            )
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens

            # Replace masked input tokens with tokenizer.mask_token_id
            indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            )
            masked_input_ids[indices_replaced] = self.tokenizer.mask_token_id

            # For the rest, replace with random words 10% of the time
            indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices & ~indices_replaced
            )
            random_words = torch.randint(
                low=0,
                high=len(self.tokenizer),
                size=labels.shape,
                dtype=torch.long
            )
            masked_input_ids[indices_random] = random_words[indices_random]

            # Return the dictionary
            return {
                'input_ids': masked_input_ids,  # Masked input ids for model input
                'labels': labels,               # Labels for computing loss
                'masked_ids': masked_indices,   # Boolean mask indicating masked tokens
                'attention_mask': attention_mask,
                'token_type_ids': encoding.get('token_type_ids'),
            }

        except Exception as e:
            raise ValueError(f"An error occurred during tokenization: {e}")
        



from typing import List, Dict, Any
import argparse
import logging
import sys
import multiprocessing
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
from functools import partial

# Assume the fixed API is already imported:
# from your_module import PromptTemplate, load_and_concat_datasets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_example(
    example: Dict[str, Any],
    prompt_template: PromptTemplate
) -> Dict[str, torch.Tensor]:
    """
    Process a single data example using the prompt template.

    Args:
        example (Dict[str, Any]): The data example to process.
        prompt_template (PromptTemplate): The prompt template instance.

    Returns:
        Dict[str, torch.Tensor]: The processed example with tokenized tensors.
    """
    try:
        processed = prompt_template(**example)
        return processed
    except Exception as e:
        logger.error(f"Error processing example {example}: {e}")
        return {}  # Return empty dict to be filtered out later


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to combine processed examples into a batch.

    Args:
        batch (List[Dict[str, torch.Tensor]]): List of processed examples.

    Returns:
        Dict[str, torch.Tensor]: Batched tensors.
    """
    batch = [item for item in batch if item]  # Filter out any failed processing
    if not batch:
        return {}
    keys = batch[0].keys()
    collated = {key: torch.stack([item[key] for item in batch]) for key in keys}
    return collated


def preprocess_data(
    dataset_names: List[str],
    template_str: str,
    tokenizer_name: str,
    batch_size: int = 32,
    num_workers: int = multiprocessing.cpu_count(),
    distributed: bool = False
) -> DataLoader:
    """
    Preprocess data from multiple datasets and return a DataLoader.

    Args:
        dataset_names (List[str]): Names of the datasets to load.
        template_str (str): The prompt template string.
        tokenizer_name (str): The name or path of the tokenizer.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        num_workers (int, optional): Number of worker processes. Defaults to CPU count.
        distributed (bool, optional): Use distributed data parallelism. Defaults to False.

    Returns:
        DataLoader: DataLoader yielding batches of processed data.
    """
    try:
        # Load and concatenate datasets
        concatenated_dataset = load_and_concat_datasets(dataset_names)
        logger.info("Datasets loaded and concatenated.")

        # Initialize the prompt template
        prompt_template = PromptTemplate(
            tokenizer_name=tokenizer_name,
            template=template_str
        )
        logger.info("PromptTemplate initialized.")

        # Define a PyTorch Dataset wrapper
        class ProcessedDataset(TorchDataset):
            def __init__(self, dataset: TorchDataset) -> None:
                self.dataset = dataset

            def __len__(self) -> int:
                return len(self.dataset)

            def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
                example = self.dataset[index]
                processed_example = process_example(example, prompt_template)
                return processed_example

        processed_dataset = ProcessedDataset(concatenated_dataset)
        logger.info("ProcessedDataset created.")

        if distributed:
            if not dist.is_initialized():
                raise RuntimeError("Distributed processing requested but process group is not initialized.")
            sampler = DistributedSampler(processed_dataset)
            shuffle = False
            logger.info("Using DistributedSampler for distributed processing.")
        else:
            sampler = None
            shuffle = True  # Shuffle for non-distributed training

        # Create DataLoader with multiprocessing and optional distributed sampler
        data_loader = DataLoader(
            dataset=processed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True
        )
        logger.info("DataLoader created.")

        return data_loader

    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        raise


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument(
        "--dataset_names", nargs="+", required=True,
        help="List of dataset names to preprocess."
    )
    parser.add_argument(
        "--template_str", type=str, required=True,
        help="Prompt template string with placeholders."
    )
    parser.add_argument(
        "--tokenizer_name", type=str, required=True,
        help="Name or path of the tokenizer to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for the DataLoader."
    )
    parser.add_argument(
        "--num_workers", type=int, default=multiprocessing.cpu_count(),
        help="Number of worker processes for data loading."
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed data parallel processing."
    )
    parser.add_argument(
        "--backend", type=str, default="nccl",
        help="Backend for torch distributed (e.g., 'nccl', 'gloo')."
    )
    args = parser.parse_args()

    if args.distributed:
        dist.init_process_group(backend=args.backend)
        logger.info("Distributed process group initialized.")

    try:
        # Preprocess data and get the DataLoader
        data_loader = preprocess_data(
            dataset_names=args.dataset_names,
            template_str=args.template_str,
            tokenizer_name=args.tokenizer_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=args.distributed
        )

        # Iterate over DataLoader
        for batch in data_loader:
            if not batch:
                continue  # Skip empty batches
            # Here you can pass the batch to your model for training or inference
            # For example: outputs = model(**batch)
            pass

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
    finally:
        if args.distributed:
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed.")



#  pip install torch transformers datasets

#    - If using distributed processing, ensure you have set up the environment correctly (e.g., multiple GPUs).

# 2. **Run the Script**:
#    - Basic usage without distributed processing:

#      python preprocess.py \
#        --dataset_names squad imdb \
#        --template_str "Question: {question} Answer: {answer}" \
#        --tokenizer_name bert-base-uncased \
#        --batch_size 64 \
#        --num_workers 4

#    - With distributed processing:

#      python -m torch.distributed.launch --nproc_per_node=NUM_GPUS preprocess.py \
#        --dataset_names squad imdb \
#        --template_str "Question: {question} Answer: {answer}" \
#        --tokenizer_name bert-base-uncased \
#        --batch_size 64 \
#        --num_workers 4 \
#        --distributed \
#        --backend nccl

#      Replace `NUM_GPUS` with the number of GPUs available.

# **Notes**:

# - **Template String**: Ensure that the placeholders in `template_str` match the keys in the datasets you are loading.
# - **Tokenizer**: Replace `bert-base-uncased` with the tokenizer appropriate for your use case.
# - **Distributed Backend**: The default backend is set to `nccl`, which is optimal for NVIDIA GPUs. For CPU-only environments, you might need to use `gloo`.

# **Adjustments**:

# - You can modify the logging level or format as needed.
# - Extend `process_example` or `collate_fn` to include additional processing steps.

# This preprocessing code is designed to be robust, handle large datasets efficiently, and integrate seamlessly with PyTorch's distributed training capabilities.