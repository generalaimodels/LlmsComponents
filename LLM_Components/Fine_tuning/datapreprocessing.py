from typing import Dict, List
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import logging
import copy

import torch
from torch import Tensor
import pandas as pd
from transformers import PreTrainedTokenizer

from config import DataproConfig,DatasetConfig
from Dataloader import DatasetLoader


logger = logging.getLogger(__name__)






class PromptTemplate:
    """Manages the template for formatting prompts."""

    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        try:
            return self.template.format(**{k: kwargs.get(k, '') for k in self.input_variables})
        except KeyError as e:
            logger.error(f"Missing key in prompt formatting: {e}")
            raise


class DatasetProcessor:
    """Process datasets for model training or evaluation."""

    def __init__(self, 
                 dataloader_config: DatasetConfig, 
                 datasetpro_config: DataproConfig, 
                 tokenizer: PreTrainedTokenizer, 
                 prompt_template: PromptTemplate):
        self.config = datasetpro_config
        self.datasetloader_config = dataloader_config
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

    def load_and_split_dataset(self) -> DatasetDict:
        try:
            dataset = DatasetLoader(config=self.datasetloader_config).load()
            return dataset.train_test_split(test_size=self.config.eval_ratio + self.config.test_ratio, 
                                           shuffle=True, 
                                           seed=42)
        except Exception as e:
            logger.exception(f"Failed to load or split the dataset: {e}")
            raise

    def validate_columns(self, dataset: Dataset):
        required_columns = self.config.input_columns + [self.config.target_column]
        missing = set(required_columns) - set(dataset.column_names)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def process_dataset(self) -> DatasetDict:
        try:
            dataset_dict = self.load_and_split_dataset()
            for split in dataset_dict:
                self.validate_columns(dataset_dict[split])
                dataset_dict[split] = dataset_dict[split].map(
                    self.apply_prompt_template, batched=True, remove_columns=dataset_dict[split].column_names
                ).map(
                    self.tokenize_and_add_labels, batched=True, remove_columns=["prompt", "target"]
                )
            return dataset_dict
        except Exception as e:
            logger.error(f"Dataset processing error: {e}")
            raise

    def apply_prompt_template(self, batch: Dict[str, List]) -> Dict[str, List]:
        return {
            "prompt": [self.prompt_template.format(**{col: text for col, text in zip(self.config.input_columns, texts)})
                       for texts in zip(*(batch[col] for col in self.config.input_columns))],
            "target": batch[self.config.target_column]
        }

    def tokenize_and_add_labels(self, batch: Dict[str, List]) -> Dict[str, List]:
        result = {"input_ids": [], "attention_mask": [], "labels": [], "label_g": []}
        for prompt, target in zip(batch["prompt"], batch["target"]):
            encoded = self._encode_pair(prompt, target)
            for key, value in encoded.items():
                result[key].append(value)
        return result

    def _encode_pair(self, prompt: str, target: str) -> Dict:
        encoded_prompt = self.tokenizer.encode(
            self.tokenizer.bos_token + prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length // 2,
        )
        encoded_target = self.tokenizer.encode(
            target + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length - len(encoded_prompt),
        )

        combined = encoded_prompt + encoded_target
        padding = self.config.max_length - len(combined)
        attention_mask = [1] * len(combined) + [0] * padding
        
        return {
            "input_ids": combined + [self.tokenizer.pad_token_id] * padding,
            "attention_mask": attention_mask,
            "labels": [-100] * len(encoded_prompt) + encoded_target + [-100] * padding,
            "label_g": copy.deepcopy(combined + [self.tokenizer.pad_token_id] * padding),
        }
        



class AdvancedTokenizer:
    """
    
    Example usage:
    tokenizer = ...  # Your pretrained tokenizer
    special_tokens = ["<bos>", "<eos>", "<unk>", "<spe>", "<role>", "<instruction>"]
    advanced_tokenizer = AdvancedTokenizer(tokenizer, special_tokens)
    
    df = pd.DataFrame(...)  # Your input dataframe
    column_names = ["col1", "col2"]
    target_names = ["target"]
    
    result = advanced_tokenizer.tokenize(df, column_names, target_names)
    
    # To decode:
    decoded_text = advanced_tokenizer.decode(result["input_ids"])

    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        special_tokens: List[str],
        max_length: int = 512,
        batch_size: int = 32
    ):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_length = max_length
        self.batch_size = batch_size
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Add special tokens to the tokenizer."""
        special_tokens_dict = {"additional_special_tokens": self.special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def _validate_input(
        self, data: pd.DataFrame, column_names: List[str], target_names: List[str]
    ) -> None:
        """Validate input data and column names."""
        if data.empty or not column_names or not target_names:
            raise ValueError("Invalid or empty input data.")
        if not set(column_names + target_names).issubset(data.columns):
            raise ValueError("Specified columns not found in the dataframe.")

    def _prepare_sequences(
        self, data: pd.DataFrame, column_names: List[str]
    ) -> List[str]:
        """Prepare input or target sequences with special tokens."""
        sequences = data[column_names].agg(' '.join, axis=1).tolist()
        return [
            f"{self.tokenizer.bos_token} {seq} {self.tokenizer.eos_token}" 
            for seq in sequences
        ]

    def _batch_tokenize(
        self,
        input_sequences: List[str],
        target_sequences: List[str],
    ) -> Dict[str, Tensor]:
        """Tokenize sequences in batches for memory efficiency."""
        total_samples = len(input_sequences)
        input_ids, attention_masks, labels, label_gs = [], [], [], []

        for i in range(0, total_samples, self.batch_size):
            batch_input = input_sequences[i:i+self.batch_size]
            batch_target = target_sequences[i:i+self.batch_size]

            inputs = self.tokenizer(
                batch_input,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            targets = self.tokenizer(
                batch_target,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            batch_labels = targets['input_ids'].clone()
            batch_labels[batch_labels == self.tokenizer.pad_token_id] = -100

            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
            labels.append(batch_labels)
            label_gs.append(inputs['input_ids'].clone())

        return {
            "input_ids": torch.cat(input_ids),
            "attention_mask": torch.cat(attention_masks),
            "labels": torch.cat(labels),
            "label_g": torch.cat(label_gs)
        }

    def tokenize(
        self,
        data: pd.DataFrame,
        column_names: List[str],
        target_names: List[str],
    ) -> Dict[str, Tensor]:
        """
        Process input data and return tokenized tensors with advanced features.

        Args:
            data (pd.DataFrame): Input dataframe.
            column_names (List[str]): Names of input columns.
            target_names (List[str]): Names of target columns.

        Returns:
            Dict[str, Tensor]: Dictionary containing processed tensors.

        Raises:
            ValueError: If input data is invalid or empty.
            RuntimeError: If an error occurs during processing.
        """
        try:
            self._validate_input(data, column_names, target_names)

            input_sequences = self._prepare_sequences(data, column_names)
            target_sequences = self._prepare_sequences(data, target_names)

            tokenized_data = self._batch_tokenize(input_sequences, target_sequences)

            return tokenized_data

        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}") from e

    def decode(self, token_ids: Tensor) -> List[str]:
        """
        Decode token IDs back to text.

        Args:
            token_ids (Tensor): Tensor of token IDs.

        Returns:
            List[str]: List of decoded text sequences.
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)



class AdvancedTokenizerTest:
    def __init__(
        self, 
        tokenizer, 
        special_tokens: List[str], 
        max_length: int = 512
    ):
        """
        Initialize the AdvancedTokenizer instance.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer instance.
            special_tokens (List[str]): List of special tokens.
            max_length (int): Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_length = max_length
        self._add_special_tokens()

    def _add_special_tokens(self):
        """
        Add special tokens to the tokenizer's vocabulary.
        """
        special_tokens_dict = {'additional_special_tokens': self.special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def process_data_in_batches(
        self, 
        data: pd.DataFrame, 
        column_names: List[str], 
        target_names: List[str], 
        batch_size: int
    ) -> List[Dict[str, Tensor]]:
        """
        Process input data in batches and return tokenized tensors.

        Args:
            data (pd.DataFrame): Input dataframe.
            column_names (List[str]): Names of input columns.
            target_names (List[str]): Names of target columns.
            batch_size (int): Number of samples per batch.

        Returns:
            List[Dict[str, Tensor]]: List of dictionaries containing processed tensors for each batch.
        """
        results = []
        num_samples = len(data)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data.iloc[start_idx:end_idx]
            input_sequences = self._prepare_sequences(batch_data, column_names, target_names)
            inputs = self._tokenize_sequences(input_sequences)
            label_g = inputs['input_ids'].clone()

            results.append({
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "label_g": label_g
            })
        
        return results

    def _prepare_sequences(self, data: pd.DataFrame, column_names: List[str],  target_names: List[str]) -> List[str]:
        """
        Prepare sequences by concatenating specified columns and adding special tokens.

        Args:
            data (pd.DataFrame): Input dataframe.
            column_names (List[str]): Names of columns to concatenate.
            target_names (List[str]): Names of target columns.
        Returns:
            List[str]: List of prepared sequences.
        """
        sequences_c = data[column_names].agg(' '.join, axis=1).tolist()
        sequences_t = data[target_names].agg(' '.join, axis=1).tolist()
        return [f"{self.tokenizer.bos_token} {seq_c} {seq_t} {self.tokenizer.eos_token}" for seq_c, seq_t in zip(sequences_c, sequences_t)]

    def _tokenize_sequences(self, sequences: List[str]) -> Dict[str, Tensor]:
        """
        Tokenize sequences with padding and truncation.

        Args:
            sequences (List[str]): List of sequences to tokenize.

        Returns:
            Dict[str, Tensor]: Tokenized sequences as tensors.
        """
        return self.tokenizer(
            sequences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): List of token IDs to decode.
            skip_special_tokens (bool): Whether to skip special tokens.

        Returns:
            str: Decoded string.
        """
        try:
            return self.tokenizer.decode(
                token_ids, 
                skip_special_tokens=skip_special_tokens
            )
        except Exception as e:
            raise RuntimeError(f"Error decoding tokens: {str(e)}") from e


