import torch
from typing import List, Optional, Union, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer, StoppingCriteria
import time
import logging

logger = logging.getLogger(__name__)

class CustomBatchStoppingCriteria(StoppingCriteria):
    """
    Custom `StoppingCriteria` which checks if all generations in the batch are completed.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        eos_token_ids: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        max_time: Optional[float] = None,
        min_length: int = 0,
        max_repeated_ngrams: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.eos_token_ids = eos_token_ids or [tokenizer.eos_token_id]
        self.stop_strings = stop_strings or []
        self.start_time = time.time()
        self.max_time = max_time
        self.max_repeated_ngrams = max_repeated_ngrams

        self._precompute_stop_string_data()

    def _precompute_stop_string_data(self) -> None:
        """Precompute data for efficient stop string matching."""
        self.stop_string_data = []
        for stop_string in self.stop_strings:
            stop_tokens = self.tokenizer.encode(stop_string, add_special_tokens=False)
            self.stop_string_data.append({
                'tokens': stop_tokens,
                'length': len(stop_tokens),
            })

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if all sequences in the batch meet the stopping criteria."""
        if self.max_time and (time.time() - self.start_time) > self.max_time:
            logger.info("Stopping due to max time limit")
            return True

        batch_size, seq_length = input_ids.shape

        if seq_length < self.min_length:
            return False

        if seq_length >= self.max_length:
            logger.info("Stopping due to max length")
            return True

        if self.eos_token_ids:
            last_token_is_eos = input_ids[:, -1].unsqueeze(-1) == torch.tensor(self.eos_token_ids, device=input_ids.device)
            if last_token_is_eos.any():
                logger.info("Stopping due to EOS token generation")
                return True

        if self.stop_strings:
            for stop_data in self.stop_string_data:
                stop_length = stop_data['length']
                if seq_length >= stop_length:
                    last_tokens = input_ids[:, -stop_length:]
                    matches = torch.all(last_tokens == torch.tensor(stop_data['tokens'], device=input_ids.device), dim=1)
                    if matches.any():
                        logger.info("Stopping due to stop string match")
                        return True

        if self.max_repeated_ngrams:
            if self._check_repeated_ngrams(input_ids):
                logger.info("Stopping due to repeated n-grams")
                return True

        return False

    def _check_repeated_ngrams(self, input_ids: torch.LongTensor) -> bool:
        """Check if there are repeated n-grams exceeding the maximum allowed repetitions."""
        if self.max_repeated_ngrams is None:
            return False

        batch_size, seq_length = input_ids.shape
        for n in range(1, min(seq_length // 2, 4) + 1):  # Check n-grams up to 4 or half the sequence length
            for i in range(seq_length - n):
                ngram = input_ids[:, i:i+n]
                remaining = input_ids[:, i+n:]
                matches = (remaining == ngram.unsqueeze(1)).all(dim=2).any(dim=1)
                if matches.sum() > self.max_repeated_ngrams:
                    return True
        return False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CustomBatchStoppingCriteria':
        """Create a CustomBatchStoppingCriteria instance from a configuration dictionary."""
        required_params = ['tokenizer', 'max_length']
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")

        return cls(
            tokenizer=config['tokenizer'],
            max_length=config['max_length'],
            eos_token_ids=config.get('eos_token_ids'),
            stop_strings=config.get('stop_strings'),
            max_time=config.get('max_time'),
            min_length=config.get('min_length', 0),
            max_repeated_ngrams=config.get('max_repeated_ngrams'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a dictionary representation."""
        return {
            'max_length': self.max_length,
            'min_length': self.min_length,
            'eos_token_ids': self.eos_token_ids,
            'stop_strings': self.stop_strings,
            'max_time': self.max_time,
            'max_repeated_ngrams': self.max_repeated_ngrams,
        }

    def update(self, **kwargs: Any) -> None:
        """Update the instance attributes with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid attribute: {key}")

        if 'stop_strings' in kwargs:
            self._precompute_stop_string_data()

    @staticmethod
    def merge_criteria(criteria_list: List[StoppingCriteria]) -> 'CustomBatchStoppingCriteria':
        """Merge multiple stopping criteria into a single CustomBatchStoppingCriteria instance."""
        merged_config = {}
        for criteria in criteria_list:
            if isinstance(criteria, CustomBatchStoppingCriteria):
                merged_config.update(criteria.to_dict())
            else:
                logger.warning(f"Skipping incompatible criteria type: {type(criteria)}")

        if not merged_config:
            raise ValueError("No compatible criteria found to merge")

        return CustomBatchStoppingCriteria.from_config(merged_config)


def validate_stopping_criteria(stopping_criteria: Union[StoppingCriteria, List[StoppingCriteria]]) -> List[StoppingCriteria]:
    """Validate and normalize the stopping criteria."""
    if isinstance(stopping_criteria, StoppingCriteria):
        return [stopping_criteria]
    elif isinstance(stopping_criteria, list) and all(isinstance(criteria, StoppingCriteria) for criteria in stopping_criteria):
        return stopping_criteria
    else:
        raise ValueError("stopping_criteria must be a StoppingCriteria instance or a list of StoppingCriteria instances")


# Initialize GPT-2 model and tokenizer
model_name = "gpt2"
cache_dir="E:/LLMS/Fine-tuning/data"
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Set the model to evaluation mode
model.eval()

# Define custom stopping criteria parameters
max_length = 50
eos_token_ids = [tokenizer.eos_token_id]
stop_strings = ["<END>"]
max_time = 5.0  # seconds
min_length = 10
max_repeated_ngrams = 3

# Initialize custom stopping criteria
stopping_criteria = CustomBatchStoppingCriteria(
    tokenizer=tokenizer,
    max_length=max_length,
    eos_token_ids=eos_token_ids,
    stop_strings=stop_strings,
    max_time=max_time,
    min_length=min_length,
    max_repeated_ngrams=max_repeated_ngrams
)

# Define a prompt for generation
prompt = "Once upon a time"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate sequences with custom stopping criteria
outputs = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    stopping_criteria=validate_stopping_criteria(stopping_criteria),
    pad_token_id=tokenizer.eos_token_id  # Ensure proper padding
)

# Decode and print the generated sequences
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)


# Test Case 1: Validate Max Length Stopping Condition
stopping_criteria = CustomBatchStoppingCriteria(
    tokenizer=tokenizer,
    max_length=20  # Set a smaller max length for testing
)
outputs = model.generate(
    input_ids=input_ids,
    max_length=20,
    stopping_criteria=validate_stopping_criteria(stopping_criteria),
    pad_token_id=tokenizer.eos_token_id
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Max Length Test Generated Text:\n", generated_text)
assert len(tokenizer.encode(generated_text)) <= 20, "Generation did not stop at max length"

# Additional test cases can be created similarly for other stopping conditions.