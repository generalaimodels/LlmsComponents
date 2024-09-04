import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import PartialState
from accelerate.utils import gather_object

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_text(prompts: List[str], model_name: str = "microsoft/phi-2", batch_size: int = 2, max_new_tokens: int = 20) -> List[str]:
    """Generates text completions for the given prompts using a specified language model.

    Args:
        prompts (List[str]): A list of text prompts for which the model will generate completions.
        model_name (str): The name of the pre-trained model to use for text generation.
        batch_size (int): The number of prompts to process in each batch.
        max_new_tokens (int): The maximum number of new tokens to generate for each prompt.

    Returns:
        List[str]: A list of generated text completions corresponding to the input prompts.
    """

    # Initialize the distributed environment without needing the Accelerator
    distributed_state = PartialState()

    # Load model and tokenizer
    try:
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=distributed_state.device, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    pad_to_multiple_of = 8

    # Split prompts into batches
    formatted_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

    # Tokenize prompts with left padding
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        tokenized_prompts = [
            tokenizer(batch_prompts, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
            for batch_prompts in formatted_prompts
        ]
    finally:
        tokenizer.padding_side = original_padding_side

    completions_per_process = []

    # Generate text across distributed processes
    try:
        with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
            for batch in batched_prompts:
                batch = batch.to(distributed_state.device)
                outputs = model.generate(**batch, max_new_tokens=max_new_tokens)
                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                completions_per_process.extend(generated_text)
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise

    # Gather generated completions
    try:
        completions_gather = gather_object(completions_per_process)
        completions = completions_gather[:len(prompts)]
    except Exception as e:
        logger.error(f"Error during gathering results: {e}")
        raise

    return completions

if __name__ == "__main__":
    try:
        user_prompts = [
            "I would like to",
            "hello how are you",
            "what is going on",
            "roses are red and",
            "welcome to the hotel",
        ]

        # Generate and print text completions
        completions = generate_text(user_prompts)
        for i, completion in enumerate(completions):
            print(f"Prompt: {user_prompts[i]} -> Completion: {completion}")
    except Exception as e:
        logger.critical(f"An error occurred in the main execution: {e}")