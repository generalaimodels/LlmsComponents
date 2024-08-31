from typing import Dict, List
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import logging
import copy


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
        

