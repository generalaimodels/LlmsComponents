#!/usr/bin/env python
# coding=utf-8

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers import PreTrainedModel, PreTrainedTokenizer
import matplotlib.pyplot as plt

# Assuming these modules are part of your project
from customfinetuning.cconfig import DataConfig, ModelConfig, DatasetConfig
from customfinetuning.cdata import PromptTemplate, DatasetProcessorTest
from customfinetuning.cdata import ConcatDataset_batch
from customfinetuning.cmodel import ModelLoader
from training_config import Config, load_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training and evaluation script using DeepSpeed.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--deepspeed_config", type=str, required=True, help="Path to the DeepSpeed config JSON file.")
    return parser.parse_args()


def initialize_training(session_id: str, config_path: str) -> Tuple[Dict[str, Any], Any, Any, Any, Any]:
    # Load configuration
    config = Config.from_yaml(config_path, session_id)
    
    # Configurations for data, model, and prompt
    dataset_config = config.dataconfig["DataConfig"]
    data_args = config.dataconfig["DatasetConfig"]
    model_config = config.modelconfig["ModelConfig"]
    prompt_config = config.promptconfig["PromptTemplate"]
    
    # Validate and process config
    dataset_args = DataConfig(**dataset_config)
    data_args = DatasetConfig(**data_args)
    model_args = ModelConfig(**model_config)
    prompt_template = PromptTemplate(**prompt_config)
    
    return dataset_args, data_args, model_args, prompt_template, config


def load_model_and_tokenizer(model_args: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_loader = ModelLoader(model_config=model_args)
    try:
        model, tokenizer = model_loader.load_model_and_tokenizer()
    except Exception as exc:
        raise RuntimeError(f"Failed to load model and tokenizer: {exc}")
    return model, tokenizer


def prepare_dataloader(dataset_processor: DatasetProcessorTest) -> Tuple[DataLoader, DataLoader]:
    dataset_dict = dataset_processor.process_dataset()
    try:
        train_dataset = ConcatDataset_batch(dataset=dataset_dict['train'], chunk_size=100, batch_size=32)
        eval_dataset = ConcatDataset_batch(dataset=dataset_dict['test'], chunk_size=100, batch_size=32)
    except KeyError as exc:
        raise RuntimeError(f"Training or testing datasets missing: {exc}")
    return train_dataset, eval_dataset


def train(model: torch.nn.Module, optimizer: Any, scheduler: Any, train_loader: DataLoader, tokenizer: PreTrainedTokenizer, device: str):
    model.train()
    for batch in train_loader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"Training loss: {loss.item()}")


def evaluate(model: torch.nn.Module, eval_loader: DataLoader, tokenizer: PreTrainedTokenizer, device: str) -> Dict[str, float]:
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(inputs, labels=labels)
            eval_loss += outputs.loss.item()
    
    eval_loss /= len(eval_loader)
    print(f"Evaluation loss: {eval_loss}")
    
    return {"eval_loss": eval_loss}


def save_metrics(metrics: Dict[str, float], filename: str):
    with open(filename, "w") as f:
        yaml.dump(metrics, f)


def plot_metrics(metrics: Dict[str, float], filename: str):
    plt.figure()
    plt.plot(metrics["eval_loss"], label="Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Evaluation Loss Over Epochs")
    plt.legend()
    plt.savefig(filename)


def main():
    args = parse_arguments()
    
    # Load and setup configurations
    dataset_args, data_args, model_args, prompt_template, config = initialize_training("config1", args.config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # Process datasets
    dataset_processor = DatasetProcessorTest(
        dataloader_config=dataset_args,
        dataset_config=data_args,
        tokenizer=tokenizer,
        prompt_template=prompt_template
    )
    
    train_dataset, eval_dataset = prepare_dataloader(dataset_processor)
    
    # Prepare for DeepSpeed training
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train(model_engine, optimizer, scheduler, train_dataset, tokenizer, device)
    
    metrics = evaluate(model_engine, eval_dataset, tokenizer, device)
    
    save_metrics(metrics, "metrics.yaml")
    plot_metrics(metrics, "metrics.png")


if __name__ == "__main__":
    main()