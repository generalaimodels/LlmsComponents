import os
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import list_repo_files
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig, PeftConfig

from .configs import DataArguments, DPOConfig, ModelArguments, SFTConfig
from .data import DEFAULT_CHAT_TEMPLATE


def get_current_device(accelerator: Optional[Accelerator] = None) -> Union[int, str]:
    """
    Get the current device for computation.

    Parameters:
        accelerator (Optional[Accelerator]): An optional Accelerator instance. If not provided, a new one is created.

    Returns:
        Union[int, str]: The current device. Returns the local process index if CUDA is available, otherwise 'cpu'.
    """
    if accelerator is None:
        accelerator = Accelerator()
    return accelerator.local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map(device: Optional[Union[int, str]] = None) -> Optional[Dict[str, Union[int, str]]]:
    """
    Generate a device map for quantized models.

    Parameters:
        device (Optional[Union[int, str]]): The device to map to. If not provided, it defaults to the current device.

    Returns:
        Optional[Dict[str, Union[int, str]]]: A device map dictionary or None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = get_current_device()
    return {"": device}


def get_quantization_config(model_args: ModelArguments, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """
    Generate a quantization configuration based on model arguments.

    Parameters:
        model_args (ModelArguments): The model arguments containing quantization settings.
        **kwargs: Additional keyword arguments to pass to BitsAndBytesConfig.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of quantization configuration settings or None if not applicable.
    """
    quantization_config = None

    if model_args.load_in_4bit or model_args.load_in_8bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_kwargs = {
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_quant_type": model_args.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": model_args.use_bnb_nested_quant,
            "bnb_4bit_quant_storage": model_args.bnb_4bit_quant_storage,
            **kwargs,
        }

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            **quantization_kwargs
        ).to_dict()

    return quantization_config


def get_tokenizer(
    model_args: ModelArguments,
    data_args: DataArguments,
    auto_set_chat_template: bool = True,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """
    Retrieve the tokenizer based on the provided model and data arguments.

    Parameters:
        model_args (ModelArguments): The model arguments.
        data_args (DataArguments): The data arguments.
        auto_set_chat_template (bool): Automatically set the chat template if not provided.
        **kwargs: Additional keyword arguments to pass to AutoTokenizer.from_pretrained.

    Returns:
        PreTrainedTokenizerBase: The initialized tokenizer.
    """
    tokenizer_name_or_path = (
        model_args.tokenizer_name_or_path or model_args.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        **kwargs,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set default max length if the tokenizer's max length is unreasonable
    if tokenizer.model_max_length and tokenizer.model_max_length > data_args.model_max_length_threshold:
        tokenizer.model_max_length = data_args.default_model_max_length

    if data_args.chat_template:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and getattr(tokenizer, 'get_chat_template', lambda: None)() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def get_peft_config(model_args: ModelArguments, **kwargs: Any) -> Optional[PeftConfig]:
    """
    Generate a PEFT configuration based on model arguments.

    Parameters:
        model_args (ModelArguments): The model arguments containing PEFT settings.
        **kwargs: Additional keyword arguments to pass to LoraConfig.

    Returns:
        Optional[PeftConfig]: A PEFT configuration or None if PEFT is not used.
    """
    if not model_args.use_peft:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias=model_args.lora_bias or "none",
        task_type=model_args.lora_task_type or "CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
        **kwargs,
    )
    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    """
    Check if the given model path is an adapter model.

    Parameters:
        model_name_or_path (str): The model name or local path.
        revision (str): The model revision.

    Returns:
        bool: True if the model is an adapter model, False otherwise.
    """
    try:
        # Attempt to list files from the repository on HuggingFace Hub
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError, RepositoryNotFoundError):
        # If not found, check the local directory
        repo_files = os.listdir(model_name_or_path)

    adapter_files = {"adapter_model.safetensors", "adapter_model.bin"}
    return not adapter_files.isdisjoint(repo_files)


def get_checkpoint(
    training_args: Union[SFTConfig, DPOConfig],
    **kwargs: Any
) -> Optional[Path]:
    """
    Retrieve the last checkpoint from the output directory if it exists.

    Parameters:
        training_args (Union[SFTConfig, DPOConfig]): The training configuration arguments.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Optional[Path]: The path to the last checkpoint or None if not found.
    """
    if os.path.isdir(training_args.output_dir):
        last_checkpoint_path = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint_path:
            return Path(last_checkpoint_path)
    return None