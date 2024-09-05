# coding=utf-8
import logging
import copy
from typing import Dict, List, Any
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import copy
from typing import List, Dict, Any
from enum import Enum
from .data_loader import DatasetLoader
from cfinetuning.cconfig import DataConfig, DatasetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]) -> None:
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**{k: kwargs.get(k, '') for k in self.input_variables})

class DatasetProcessorTest:
    def __init__(
        self,
        dataloader_config: DataConfig,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        prompt_template: PromptTemplate,
    ) -> None:
        self.config = dataset_config
        self.datasetloader_config = dataloader_config
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

        # Ensure the tokenizer knows about the special token
        self.special_token_for_none = "[NONE]"
        if self.special_token_for_none not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.special_token_for_none])
            logger.info(f"Added special token for none: {self.special_token_for_none}")

    def load_and_split_dataset(self) -> DatasetDict:
        try:
            dataset = DatasetLoader(config=self.datasetloader_config).load()
            return dataset.train_test_split(
                test_size=self.config.eval_ratio + self.config.test_ratio,
                shuffle=True,
                seed=42,
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def validate_columns(self, dataset: Dataset) -> None:
        all_columns = self.config.input_columns + [self.config.target_column]
        missing_columns = [col for col in all_columns if col not in dataset.column_names]
        if missing_columns:
            raise ValueError(f"Missing columns in dataset: {missing_columns}")

    def apply_prompt_template(self, batch: Dict[str, List]) -> Dict[str, List]:
        prompts = []
        targets = []
        for i in range(len(batch[self.config.input_columns[0]])):
            input_dict = {col: batch[col][i] for col in self.config.input_columns}
            prompts.append(self.prompt_template.format(**input_dict))
            targets.append(batch[self.config.target_column][i])
        return {"prompt": prompts, "target": targets}

    def tokenize_and_add_labels(self, batch: Dict[str, List]) -> Dict[str, List]:
        input_ids = []
        attention_masks = []
        labels = []

        # Get the special token ID for [NONE]
        special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token_for_none)

        for prompt, target in zip(batch["prompt"], batch["target"]):
            # Use special token ID if any value is None or empty string
            prompt = prompt if prompt else self.special_token_for_none
            target = target if target else self.special_token_for_none

            # Encode prompt and target
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
            padding_length = self.config.max_length - len(combined)

            # Pad with the tokenizer's pad token ID
            input_id = combined + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(combined) + [0] * padding_length
            label = [-100] * len(encoded_prompt) + encoded_target + [-100] * padding_length

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "label_g": copy.deepcopy(input_ids)
        }

    def process_dataset(self) -> DatasetDict:
        try:
            dataset_dict = self.load_and_split_dataset()
            for split in dataset_dict:
                self.validate_columns(dataset_dict[split])
                dataset_dict[split] = (
                    dataset_dict[split]
                    .map(
                        self.apply_prompt_template,
                        batched=True,
                        batch_size=self.config.batch_size,
                        remove_columns=dataset_dict[split].column_names,
                    )
                    .map(
                        self.tokenize_and_add_labels,
                        batched=True,
                        batch_size=self.config.batch_size,
                        remove_columns=["prompt", "target"],
                    )
                )
            return dataset_dict
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise
        


class SpecialTokens(str, Enum):
    """
    Enum for managing special tokens used in tokenization and prompts.
    """
    BEGIN_TARGET = "<|begintarget|>"
    END_TARGET = "<|endtarget|>"
    BEGIN_CONTEXT = "<|begincontext|>"
    END_CONTEXT = "<|endcontext|>"
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    BEGIN_LAST_USER_UTTERANCE = "<|beginlastuserutterance|>"
    END_LAST_USER_UTTERANCE = "<|endlastuserutterance|>"
    BEGIN_DSTS = "<|begindsts|>"
    END_DSTS = "<|enddsts|>"
    BEGIN_DST = "<|begindst|>"
    END_DST = "<|enddst|>"
    BEGIN_BELIEF = "<|beginbelief|>"
    END_BELIEF = "<|endbelief|>"
    BEGIN_RESPONSE = "<|beginresponse|>"
    END_RESPONSE = "<|endresponse|>"
    BEGIN_ACTION = "<|beginaction|>"
    END_ACTION = "<|endaction|>"
    BEGIN_USER_ACTION = "<|beginuseraction|>"
    END_USER_ACTION = "<|enduseraction|>"
    SYS_ACTIONS = "<|sysactions|>"
    BEGIN_INTENT = "<|beginintent|>"
    END_INTENT = "<|endintent|>"
    BEGIN_REQUESTED_SLOTS = "<|beginrequestedslots|>"
    END_REQUESTED_SLOTS = "<|endrequestedslots|>"
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|startoftext|>"

    @classmethod
    def list(cls) -> List[str]:
        """
        Returns a list of all special tokens.
        """
        return [c.value for c in cls]


class TokenizerConfig:
    """
    Configuration class for the tokenizer, handling max length and validation.
    """
    def __init__(self, max_length: int, tokenizer: Any) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise TypeError("The tokenizer must have `encode` and `decode` methods.")
        self.max_length = max_length
        self.tokenizer = tokenizer


class PromptTemplate_Updated:
    """
    Handles prompt templates for construction and formatting of tokenized prompts.
    """
    def __init__(
        self,
        template: str,
        input_variables: List[str],
        config: TokenizerConfig
    ) -> None:
        if not template or not isinstance(template, str):
            raise ValueError("The template must be a non-empty string.")
        self.template = template
        self.input_variables = input_variables
        self.config = config

    def format(self, **kwargs) -> str:
        """
        Formats the prompt using the specified input variables.

        :param kwargs: Dynamic arguments matching input_variables.
        :return: A formatted string.
        :raises KeyError: If required input variables are not provided.
        """
        try:
            formatted = self.template.format(
                **{k: kwargs.get(k, '') for k in self.input_variables}
            )
            return formatted
        except KeyError as e:
            raise KeyError(f"Missing required input variable for formatting: {e}")

    def tokenize_and_add_labels(
        self,
        batch: Dict[str, List[str]]
    ) -> Dict[str, List[List[int]]]:
        """
        Tokenizes prompts and targets into input IDs, attention masks, and labels.

        :param batch: A dictionary containing 'prompt' and 'target' lists.
        :return: A dictionary with tokenized data and attention information.
        :raises RuntimeError: If tokenization fails.
        """
        input_ids, attention_masks, labels = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            try:
                encoded_prompt = self.config.tokenizer.encode(
                    self.config.tokenizer.bos_token + prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.config.max_length // 2,
                )
                encoded_target = self.config.tokenizer.encode(
                    target + self.config.tokenizer.eos_token,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.config.max_length - len(encoded_prompt),
                )

                combined = encoded_prompt + encoded_target
                padding_length = self.config.max_length - len(combined)

                input_id = combined + [self.config.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * len(combined) + [0] * padding_length
                label = (
                    [-100] * len(encoded_prompt) +
                    encoded_target +
                    [-100] * padding_length
                )

                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                labels.append(label)
            except Exception as e:
                raise RuntimeError(
                    f"Error during tokenization and label addition: {e}"
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "label_g": copy.deepcopy(input_ids),
        }


def preprocess_function(
    examples: Dict[str, List[str]],
    text_column: str,
    label_column: str,
    tokenizer: Any,
    max_length: int
) -> Dict[str, Any]:
   
    try:
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]

        model_inputs = tokenizer(
            inputs, padding=True, truncation=True, max_length=max_length // 2
        )
        labels = tokenizer(
            targets, add_special_tokens=False, truncation=True, max_length=max_length // 2
        )

        # Combine inputs and labels, handling end tokens
        model_inputs["input_ids"] = [
            input_ids + label_ids + [tokenizer.eos_token_id]
            for input_ids, label_ids in zip(model_inputs["input_ids"], labels["input_ids"])
        ]

        model_inputs["attention_mask"] = [
            [1] * len(input_ids) for input_ids in model_inputs["input_ids"]
        ]

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

            padding_needed = max_length - len(sample_input_ids)
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * padding_needed + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * padding_needed + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * padding_needed + label_input_ids

            # Trim to max_length
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
            labels["input_ids"][i] = labels["input_ids"][i][:max_length]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    except KeyError as e:
        raise KeyError(f"Missing key in input data: {e}")
    except TypeError as e:
        raise TypeError(f"Input data must be a dictionary with lists of strings: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during preprocessing: {e}")
    



 
    # Preprocesses the input and label data to tokenize and format for model input.

    # :param examples: Data dictionary containing input text and labels.
    # :param text_column: Name of the column with input texts.
    # :param label_column: Name of the column with labels to predict.
    # :param tokenizer: Tokenizer instance for encoding text.
    # :param max_length: Maximum length for token insertion.
    # :return: A dictionary containing input IDs, attention masks, and labels.
    # :raises KeyError: If required keys are missing in input data.
    # :raises TypeError: If the input data is not a dictionary with lists of strings.
    # :raises RuntimeError: For unexpected errors during preprocessing.
    # Example :
    
    
       
    #    from transformers import GPT2Tokenizer
    #    from typing import List, Dict
       
    #    # Initialize the tokenizer
    #    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=r"C:\Users\heman\Desktop\Coding\tokenizer_M")
    #    tokenizer.pad_token = SpecialTokens.PAD_TOKEN.value
    #    tokenizer.bos_token = SpecialTokens.BOS_TOKEN.value
    #    tokenizer.eos_token = SpecialTokens.END_RESPONSE.value  # Assuming END_RESPONSE as the eos_token
       
    #    # Configuring the tokenizer with a maximum length
    #    max_length = 64
    #    tokenizer_config = TokenizerConfig(max_length=max_length, tokenizer=tokenizer)
       
    #    # Define a prompt template and inputs
    #    template = "{user} {context}"
    #    input_variables = ["user", "context"]
       
    #    prompt_template = PromptTemplate_Updated(
    #        template=template,
    #        input_variables=input_variables,
    #        config=tokenizer_config,
    #    )
       
    #    # Define example inputs for testing
    #    examples = {
    #        "text_column": ["Hello, how are you?", "What is the weather today?"],
    #        "label_column": ["I'm good, thank you!", "It's sunny and warm."],
    #    }
       
    #    # Formatting inputs using PromptTemplate
    #    formatted_prompts = [
    #        prompt_template.format(user="USER:", context=text) 
    #        for text in examples["text_column"]
    #    ]
       
    #    # Tokenize and prepare batch
    #    batch = {
    #        "prompt": formatted_prompts,
    #        "target": examples["label_column"],
    #    }
       
    #    # Tokenize and add labels
    #    tokenized_data = prompt_template.tokenize_and_add_labels(batch)
       
    #    # Testing preprocess_function
    #    preprocessed_data = preprocess_function(
    #        examples={"text_column": examples["text_column"], "label_column": examples["label_column"]},
    #        text_column="text_column",
    #        label_column="label_column",
    #        tokenizer=tokenizer,
    #        max_length=max_length,
    #    )
       
    #    # Print outputs for verification
    #    print("Formatted Prompts:", formatted_prompts)
    #    print("Tokenized Data:", tokenized_data)
    #    print("Preprocessed Data:", tokenizer.decode(preprocessed_data["input_ids"][0]))
