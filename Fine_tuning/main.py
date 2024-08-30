import os
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import yaml
from tqdm import tqdm

import torch
from transformers import Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset
from huggingface_hub import HfApi

from customfinetuning.cconfig.finetuningconfig import DataConfig, ModelConfig
from customfinetuning.cdata import PromptTemplate, DatasetProcessor
from customfinetuning.cdata.data_collector import ConcatDataset_batch
from customfinetuning.cmodel.modelandtokenizer import ModelLoader


@dataclass
class Config:
    data_config: DataConfig
    prompt_template: PromptTemplate
    model_config: ModelConfig
    training_args: TrainingArguments


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load configuration data from a YAML file."""
    try:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file at {path} not found.")
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error parsing YAML file: {exc}")


def process_data_config(config_data: Dict[str, Any]) -> DataConfig:
    """Create DataConfig from configuration data."""
    try:
        return DataConfig(**config_data)
    except TypeError as exc:
        raise ValueError(f"Incorrect data configuration: {exc}")


def process_model_config(config_data: Dict[str, Any]) -> ModelConfig:
    """Create ModelConfig from configuration data."""
    try:
        return ModelConfig(**config_data)
    except TypeError as exc:
        raise ValueError(f"Incorrect model configuration: {exc}")


def process_prompt_template(config_data: Dict[str, Any]) -> PromptTemplate:
    """Create PromptTemplate from configuration data."""
    try:
        return PromptTemplate(**config_data)
    except KeyError as exc:
        raise ValueError(f"Missing key in prompt template configuration: {exc}")


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_model(
    model: torch.nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Train the model and return training and evaluation results."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    return train_result.metrics, eval_result


def save_results(
    train_results: Dict[str, float],
    eval_results: Dict[str, float],
    output_dir: str
) -> None:
    """Save training and evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train_results.txt"), "w") as f:
        for key, value in train_results.items():
            f.write(f"{key}: {value}\n")
    
    with open(os.path.join(output_dir, "eval_results.txt"), "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")


def push_to_hub(model: torch.nn.Module, tokenizer: Any, repo_name: str) -> None:
    """Push the model and tokenizer to Hugging Face Hub."""
    api = HfApi()
    try:
        api.create_repo(repo_name, exist_ok=True)
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
    except Exception as e:
        raise RuntimeError(f"Failed to push to Hugging Face Hub: {e}")


def main(config_path: str):
    """Main function to execute the script based on the provided configuration."""
    config_data = load_yaml_config(config_path)
    
    data_config = process_data_config(config_data['data_config'])
    model_config = process_model_config(config_data['model_config'])
    prompt_template = process_prompt_template(config_data['prompt_template'])
    training_args = TrainingArguments(**config_data['training_args'])

    model_loader = ModelLoader(model_config=model_config)
    
    try:
        model, tokenizer = model_loader.load_model_and_tokenizer()
    except Exception as exc:
        raise RuntimeError(f"Failed to load model and tokenizer: {exc}")

    try:
        dataset_processor = DatasetProcessor(config=data_config, tokenizer=tokenizer, prompt_template=prompt_template)
        dataset_dict = dataset_processor.process_dataset()
    except Exception as exc:
        raise RuntimeError(f"Dataset processing failed: {exc}")

    try:
        train_dataset = ConcatDataset_batch(dataset=dataset_dict['train'], chunk_size=512, batch_size=2)
        eval_dataset = ConcatDataset_batch(dataset=dataset_dict['test'], chunk_size=512, batch_size=2)
    except KeyError as exc:
        raise RuntimeError(f"Training or testing datasets missing: {exc}")

    # Training and evaluation
    train_results, eval_results = train_model(model, train_dataset, eval_dataset, training_args)

    # Save results
    save_results(train_results, eval_results, training_args.output_dir)

    # Save model weights locally
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Push to Hugging Face Hub
    push_to_hub(model, tokenizer, config_data['hub_repo_name'])

    print("Training completed. Results and model saved locally and pushed to Hugging Face Hub.")


if __name__ == "__main__":
    try:
        main("config.yml")
    except Exception as e:
        print(f"An error occurred: {e}")