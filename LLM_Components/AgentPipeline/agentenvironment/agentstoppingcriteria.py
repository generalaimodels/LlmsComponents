from typing import List, Optional, Union
import time
import torch
from transformers import PreTrainedTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList



class CustomBatchStoppingCriteria(StoppingCriteria):
    """
    Custom `StoppingCriteria` which checks if all generations in the batch are completed.

    This class provides a flexible and efficient way to determine when to stop generation
    for all sequences in a batch, based on various criteria such as maximum length,
    end-of-sequence tokens, or custom stop strings.

    Attributes:
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing input and output.
        max_length (int): The maximum allowed length for generated sequences.
        eos_token_ids (List[int]): List of end-of-sequence token IDs.
        stop_strings (List[str]): List of stop strings that signal the end of generation.
        start_time (float): The time when generation started.
        max_time (Optional[float]): The maximum allowed time for generation in seconds.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum length of generated sequences.
        eos_token_ids (Optional[List[int]]): List of end-of-sequence token IDs.
        stop_strings (Optional[List[str]]): List of stop strings.
        max_time (Optional[float]): Maximum allowed time for generation in seconds.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        eos_token_ids: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        max_time: Optional[float] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_ids = eos_token_ids or [tokenizer.eos_token_id]
        self.stop_strings = stop_strings or []
        self.start_time = time.time()
        self.max_time = max_time

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
        """
        Check if all sequences in the batch meet the stopping criteria.

        Args:
            input_ids (torch.LongTensor): The current input IDs of shape (batch_size, sequence_length).
            scores (torch.FloatTensor): The current scores of shape (batch_size, vocab_size).

        Returns:
            bool: True if all sequences in the batch should stop generating, False otherwise.
        """
        if self.max_time and (time.time() - self.start_time) > self.max_time:
            return True

        batch_size, seq_length = input_ids.shape

        # Check max length
        if seq_length >= self.max_length:
            return True

        # Check for EOS tokens
        if self.eos_token_ids:
            last_token_is_eos = input_ids[:, -1].unsqueeze(-1) == torch.tensor(self.eos_token_ids, device=input_ids.device)
            if last_token_is_eos.any():
                return True

        # Check for stop strings
        if self.stop_strings:
            for stop_data in self.stop_string_data:
                stop_length = stop_data['length']
                if seq_length >= stop_length:
                    last_tokens = input_ids[:, -stop_length:]
                    matches = torch.all(last_tokens == torch.tensor(stop_data['tokens'], device=input_ids.device), dim=1)
                    if matches.any():
                        return True

        return False

    @classmethod
    def from_config(cls, config: dict) -> 'CustomBatchStoppingCriteria':
        """
        Create a CustomBatchStoppingCriteria instance from a configuration dictionary.

        Args:
            config (dict): A dictionary containing the configuration parameters.

        Returns:
            CustomBatchStoppingCriteria: An instance of the class.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
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
        )


def validate_stopping_criteria(stopping_criteria: Union[StoppingCriteria, List[StoppingCriteria]]) -> List[StoppingCriteria]:
    """
    Validate and normalize the stopping criteria.

    Args:
        stopping_criteria (Union[StoppingCriteria, List[StoppingCriteria]]): The stopping criteria to validate.

    Returns:
        List[StoppingCriteria]: A list of validated stopping criteria.

    Raises:
        ValueError: If the input is not a StoppingCriteria instance or a list of StoppingCriteria instances.
    """
    if isinstance(stopping_criteria, StoppingCriteria):
        return [stopping_criteria]
    elif isinstance(stopping_criteria, list) and all(isinstance(criteria, StoppingCriteria) for criteria in stopping_criteria):
        return stopping_criteria
    else:
        raise ValueError("stopping_criteria must be a StoppingCriteria instance or a list of StoppingCriteria instances")
    


import time
from typing import Optional, Union, List
from transformers import PreTrainedTokenizer
from transformers import StoppingCriteria
import torch


class BatchCompletionStoppingCriteria(StoppingCriteria):
    def __init__(self, batch_size: int, max_length: Optional[int] = None):
        """
        Custom stopping criteria which checks if all generations in the batch are completed.
        
        Args:
            batch_size (int): Number of sequences in the batch.
            max_length (int, optional): Maximum length of the sequence to stop generation.
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.completed_sequences = [False] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None) -> bool:
        """
        Checks if generation should stop based on criteria.
        
        Args:
            input_ids (torch.LongTensor): The generated token ids.
            scores (Optional[torch.FloatTensor]): The scores of generated tokens.
            
        Returns:
            bool: True if all sequences in the batch are completed, False otherwise.
        """
        for i in range(input_ids.shape[0]):
            if self.max_length and input_ids.shape[1] >= self.max_length:
                self.completed_sequences[i] = True

        return all(self.completed_sequences)


class CustomTokenizerStopStringCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, stop_strings: Union[str, List[str]]):
        """
        Stop generation when any of the specified stop strings are generated.
        
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer used to decode sequences.
            stop_strings (Union[str, List[str]]): List of stop strings to end generation.
        """
        self.tokenizer = tokenizer
        if isinstance(stop_strings, str):
            self.stop_strings = [stop_strings]
        else:
            self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None) -> bool:
        """
        Checks if generation should stop based on stop strings.
        
        Args:
            input_ids (torch.LongTensor): The generated token ids.
            scores (Optional[torch.FloatTensor]): The scores of generated tokens.
            
        Returns:
            bool: True if any stop string condition is met, False otherwise.
        """
        decoded_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        for stop_string in self.stop_strings:
            if stop_string in decoded_output:
                return True

        return False


# # Example usage:
# if __name__ == "__main__":
#     from transformers import GPT2Tokenizer

#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2",cache_dir=r"E:\LLMS\Fine-tuning\data")
#     stop_strings = ["<END>", "STOP"]
#     max_length = 50
#     batch_size = 3

#     # Example input_ids for demonstration (normally generated by a model)
#     input_ids = torch.tensor([
#         tokenizer.encode("This is a test sequence", return_tensors="pt").squeeze(0),
#         tokenizer.encode("Another test sequence that is longer than the first one", return_tensors="pt").squeeze(0),
#         tokenizer.encode("Short sequence", return_tensors="pt").squeeze(0)
#     ])

#     # Initialize stopping criteria
#     batch_completion_criteria = BatchCompletionStoppingCriteria(batch_size=batch_size, max_length=max_length)
#     stop_string_criteria = CustomTokenizerStopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)

#     # Simulate a generation process
#     max_iterations = 100
#     for _ in range(max_iterations):
#         # Simulate generation step (appending new tokens to input_ids)
#         # Here we just repeat the input_ids for demonstration; typically, you'd add new tokens from model output.
#         generated_ids = torch.cat([input_ids, input_ids[:, :1]], dim=1)

#         # Check stopping criteria
#         if batch_completion_criteria(generated_ids) or stop_string_criteria(generated_ids):
#             print("Stopping criteria met. Ending generation.")
#             break

#     print("Final generated sequences:")
#     for seq in generated_ids:
#         print(tokenizer.decode(seq, skip_special_tokens=True))



import torch
from typing import List, Optional, Union, Dict, Any
from transformers import PreTrainedTokenizer
from transformers import StoppingCriteria
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




import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize GPT-2 model and tokenizer
model_name = "gpt2"
cache_dir="E:\LLMS\Fine-tuning\data"
model = GPT2LMHeadModel.from_pretrained(model_name,cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_name,cache_dir=cache_dir)

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
