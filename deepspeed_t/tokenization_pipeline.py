import copy
from typing import List, Dict, Any
from enum import Enum

class SpecialTokens(str, Enum):
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
        return [c.value for c in cls]

class TokenizerConfig:
    def __init__(self, max_length: int, tokenizer: Any):
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise TypeError("The tokenizer must have `encode` and `decode` methods.")
        self.max_length = max_length
        self.tokenizer = tokenizer

class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str], config: TokenizerConfig):
        if not template or not isinstance(template, str):
            raise ValueError("The template must be a non-empty string.")
        self.template = template
        self.input_variables = input_variables
        self.config = config

    def format(self, **kwargs) -> str:
        try:
            formatted = self.template.format(**{k: kwargs.get(k, '') for k in self.input_variables})
            return formatted
        except KeyError as e:
            raise KeyError(f"Missing required input variable for formatting: {e}")

    def tokenize_and_add_labels(self, batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
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
                label = [-100] * len(encoded_prompt) + encoded_target + [-100] * padding_length

                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                labels.append(label)
            except Exception as e:
                raise RuntimeError(f"Error during tokenization and label addition: {e}")

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

        model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=max_length // 2)
        labels = tokenizer(targets, add_special_tokens=False, truncation=True, max_length=max_length // 2)

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

            model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
            labels["input_ids"][i] = labels["input_ids"][i][:max_length]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    except KeyError as e:
        raise KeyError(f"Missing key in input data: {e}")
    except TypeError as e:
        raise TypeError(f"Input data must be a dictionary with list of strings: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during preprocessing: {e}")