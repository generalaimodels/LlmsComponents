import os
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine

def initialize_deepspeed(
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    args: Any
) -> DeepSpeedEngine:
    """Initialize DeepSpeed for the model with optimizer and lr_scheduler."""
    config = {
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": {
            "enabled": args.fp16
        },
        "zero_optimization": {
            "stage": 2  # Using ZeRO for memory optimization
        }
    }
    
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=config
    )
    
    return model

def save_checkpoint(engine: DeepSpeedEngine, checkpoint_dir: str, epoch: int) -> None:
    """Save the model checkpoint at the specified directory."""
    checkpoint_prefix = f'checkpoint_epoch_{epoch}'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_prefix)
    engine.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}.")

def load_checkpoint(model: nn.Module, checkpoint_dir: str, tag: str) -> None:
    """Load the model checkpoint if it exists."""
    try:
        _, client_state = model.load_checkpoint(checkpoint_dir, tag=tag)
        print(f"Checkpoint loaded from {checkpoint_dir} at {tag}.")
    except Exception as e:
        print(f"Failed to load checkpoint. Error: {str(e)}")

def train(
    engine: DeepSpeedEngine,
    train_loader: DataLoader,
    epoch: int
) -> None:
    """Training loop using DeepSpeed."""
    engine.train()
    for step, batch in enumerate(train_loader):
        inputs, labels = batch

        # Forward pass
        outputs = engine(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass
        engine.backward(loss)
        engine.step()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

def evaluate(
    engine: DeepSpeedEngine,
    test_loader: DataLoader
) -> None:
    """Evaluation loop using DeepSpeed."""
    engine.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            inputs, labels = batch

            # Forward pass
            outputs = engine(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_loss}, Accuracy: {accuracy}")

def main(args: Any) -> None:
    # Loading model, optimizer, scheduler, dataloaders here
    model = ...  # Your model
    optimizer = ...  # Your optimizer
    scheduler = ...  # Your learning rate scheduler
    train_loader = ...  # Your training DataLoader
    test_loader = ...  # Your testing DataLoader

    # Initialize DeepSpeed
    engine = initialize_deepspeed(model, optimizer, scheduler, args)

    # Load checkpoint if necessary
    load_checkpoint(engine, args.checkpoint_resume, tag='latest')

    # Training and evaluation loop
    for epoch in range(args.epochs):
        train(engine, train_loader, epoch)
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(engine, args.checkpoint_resume, epoch)
        
        evaluate(engine, test_loader)

    # Final model save
    save_checkpoint(engine, args.checkpoint_resume, epoch='final')

# Replace `args` with an appropriate argument parser or configuration method
# that provides necessary configurations like batch size, epochs, etc.

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_deepspeed(model: nn.Module, optimizer: Optimizer, args: Dict[str, Any]) -> deepspeed.DeepSpeedEngine:
    """
    Configure DeepSpeed for the model.

    Args:
        model: The PyTorch model to be configured.
        optimizer: The optimizer to be used.
        args: Dictionary containing DeepSpeed configuration.

    Returns:
        DeepSpeed engine.
    """
    try:
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=args
        )
        return model_engine
    except Exception as e:
        logger.error(f"Failed to initialize DeepSpeed: {str(e)}")
        raise


def save_checkpoint(model_engine: deepspeed.DeepSpeedEngine, path: str, epoch: int) -> None:
    """
    Save a checkpoint of the model.

    Args:
        model_engine: The DeepSpeed engine.
        path: Path to save the checkpoint.
        epoch: Current epoch number.
    """
    try:
        checkpoint_path = os.path.join(path, f"checkpoint_epoch_{epoch}")
        model_engine.save_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")


def load_checkpoint(model_engine: deepspeed.DeepSpeedEngine, path: str) -> Optional[int]:
    """
    Load a checkpoint of the model.

    Args:
        model_engine: The DeepSpeed engine.
        path: Path to load the checkpoint from.

    Returns:
        The epoch number of the loaded checkpoint, or None if no checkpoint was loaded.
    """
    try:
        _, client_state = model_engine.load_checkpoint(path)
        epoch = client_state['epoch']
        logger.info(f"Checkpoint loaded from {path}, resuming from epoch {epoch}")
        return epoch
    except FileNotFoundError:
        logger.info("No checkpoint found, starting from scratch")
        return None
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return None


def train_epoch(model_engine: deepspeed.DeepSpeedEngine,
                train_loader: DataLoader,
                criterion: nn.Module,
                epoch: int) -> float:
    """
    Train the model for one epoch.

    Args:
        model_engine: The DeepSpeed engine.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        epoch: Current epoch number.

    Returns:
        Average loss for the epoch.
    """
    model_engine.train()
    total_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as tepoch:
        for batch in tepoch:
            inputs, targets = batch
            inputs, targets = inputs.to(model_engine.device), targets.to(model_engine.device)

            outputs = model_engine(inputs)
            loss = criterion(outputs, targets)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)


def evaluate(model_engine: deepspeed.DeepSpeedEngine,
             test_loader: DataLoader,
             criterion: nn.Module) -> Dict[str, float]:
    """
    Evaluate the model on the test set.

    Args:
        model_engine: The DeepSpeed engine.
        test_loader: DataLoader for test data.
        criterion: Loss function.

    Returns:
        Dictionary containing evaluation metrics.
    """
    model_engine.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", unit="batch"):
            inputs, targets = inputs.to(model_engine.device), targets.to(model_engine.device)
            outputs = model_engine(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return {"accuracy": accuracy, "loss": avg_loss}


def main(model: nn.Module,
         train_loader: DataLoader,
         test_loader: DataLoader,
         optimizer: Optimizer,
         scheduler: _LRScheduler,
         criterion: nn.Module,
         checkpoint_resume: str,
         outfolder: str,
         num_epochs: int,
         deepspeed_config: Dict[str, Any]) -> None:
    """
    Main training and evaluation loop.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        optimizer: The optimizer to use.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        checkpoint_resume: Path to resume checkpoint from.
        outfolder: Folder to save outputs.
        num_epochs: Number of epochs to train.
        deepspeed_config: DeepSpeed configuration dictionary.
    """
    try:
        os.makedirs(outfolder, exist_ok=True)

        model_engine = configure_deepspeed(model, optimizer, deepspeed_config)

        start_epoch = load_checkpoint(model_engine, checkpoint_resume) or 0

        for epoch in range(start_epoch, num_epochs):
            train_loss = train_epoch(model_engine, train_loader, criterion, epoch)
            eval_metrics = evaluate(model_engine, test_loader, criterion)

            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                        f"Test Loss: {eval_metrics['loss']:.4f}, "
                        f"Test Accuracy: {eval_metrics['accuracy']:.2f}%")

            scheduler.step()

            save_checkpoint(model_engine, outfolder, epoch)

        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

import os
import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, Tuple

# Define a function to configure and load the model with DeepSpeed
def configure_model_and_optimizer(model_name: str, learning_rate: float) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return tokenizer, model, optimizer

# Training loop function
def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    deepspeed_engine: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_dir: str,
    max_epochs: int
):
    model.train()
    global_step = 0
    
    for epoch in range(max_epochs):
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(deepspeed_engine.local_rank)

            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            deepspeed_engine.backward(loss)
            deepspeed_engine.step()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Step: {batch_idx}, Loss: {loss.item()}')

            scheduler.step()
            global_step += 1

            # Save model checkpoints
            if global_step % 100 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{global_step}')
                deepspeed_engine.save_checkpoint(checkpoint_path)

# Evaluation loop function
def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    deepspeed_engine: Any
) -> None:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(deepspeed_engine.local_rank)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    print(f'Test Loss: {average_loss}')

# Main function
def main(
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    learning_rate: float,
    scheduler_config: Dict[str, Any],
    checkpoint_dir: str,
    max_epochs: int
):
    tokenizer, model, optimizer = configure_model_and_optimizer(model_name, learning_rate)

    # DeepSpeed Configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": scheduler_config['train_micro_batch_size_per_gpu'],
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        }
    }

    # Initialize DeepSpeed engine
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        training_data=train_loader,
        config=ds_config
    )

    # Scheduler initialization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config['step_scheduler'])

    train(
        model=model_engine,
        train_loader=train_loader,
        deepspeed_engine=model_engine,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        max_epochs=max_epochs
    )

    evaluate(
        model=model_engine,
        test_loader=test_loader,
        deepspeed_engine=model_engine
    )

# Entry point
if __name__ == "__main__":
    # Example configurations
    model_name = "gpt2"
    train_loader = ...  # Initialize your DataLoader
    test_loader = ...   # Initialize your DataLoader
    learning_rate = 5e-5
    scheduler_config = {
        'train_micro_batch_size_per_gpu': 4,
        'step_scheduler': {'step_size': 1000, 'gamma': 0.95}
    }
    checkpoint_dir = "./checkpoints"
    max_epochs = 3

    try:
        main(
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=learning_rate,
            scheduler_config=scheduler_config,
            checkpoint_dir=checkpoint_dir,
            max_epochs=max_epochs
        )
    except Exception as e:
        print(f"An error occurred: {e}")


import os
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerationModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def train_and_evaluate(
    model: TextGenerationModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    checkpoint_resume: str,
    num_epochs: int,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1
) -> None:
    """
    Train and evaluate the model using DeepSpeed.

    Args:
        model: The text generation model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        optimizer: Configured optimizer.
        scheduler: Learning rate scheduler.
        checkpoint_resume: Path to resume checkpoints.
        num_epochs: Number of training epochs.
        fp16: Whether to use mixed precision training.
        gradient_accumulation_steps: Number of steps for gradient accumulation.
    """
    try:
        # DeepSpeed configuration
        ds_config = {
            "train_batch_size": train_loader.batch_size * gradient_accumulation_steps,
            "fp16": {"enabled": fp16},
            "zero_optimization": {"stage": 2},
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-5,
                    "warmup_num_steps": 100
                }
            }
        }

        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        # Resume from checkpoint if available
        if os.path.exists(checkpoint_resume):
            _, client_state = model_engine.load_checkpoint(checkpoint_resume)
            start_epoch = client_state['epoch']
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            train_epoch(model_engine, train_loader, optimizer, scheduler, epoch)
            evaluate(model_engine, test_loader, epoch)

            # Save checkpoint
            client_state = {'epoch': epoch + 1}
            model_engine.save_checkpoint(checkpoint_resume, client_state=client_state)

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        raise

def train_epoch(
    model_engine: deepspeed.DeepSpeedEngine,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch: int
) -> None:
    """
    Train the model for one epoch.

    Args:
        model_engine: DeepSpeed model engine.
        train_loader: DataLoader for training data.
        optimizer: Configured optimizer.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.
    """
    model_engine.train()
    total_loss = 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
        for batch in tepoch:
            try:
                input_ids = batch['input_ids'].to(model_engine.device)
                attention_mask = batch['attention_mask'].to(model_engine.device)
                labels = batch['labels'].to(model_engine.device)

                outputs = model_engine(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

                model_engine.backward(loss)
                model_engine.step()
                total_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                continue

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    scheduler.step()

def evaluate(
    model_engine: deepspeed.DeepSpeedEngine,
    test_loader: DataLoader,
    epoch: int
) -> None:
    """
    Evaluate the model on the test set.

    Args:
        model_engine: DeepSpeed model engine.
        test_loader: DataLoader for testing data.
        epoch: Current epoch number.
    """
    model_engine.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                input_ids = batch['input_ids'].to(model_engine.device)
                attention_mask = batch['attention_mask'].to(model_engine.device)
                labels = batch['labels'].to(model_engine.device)

                outputs = model_engine(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.numel()

            except Exception as e:
                logger.error(f"Error in evaluation batch processing: {str(e)}")
                continue

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    logger.info(f"Epoch {epoch + 1} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

import os
from typing import Union
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam, FusedLamb
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerationModel(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def train_and_evaluate(
    model: TextGenerationModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    checkpoint_resume: str,
    num_epochs: int,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1
) -> None:
    """
    Train and evaluate the model using DeepSpeed.

    Args:
        model: The text generation model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        optimizer: Configured optimizer.
        scheduler: Learning rate scheduler.
        checkpoint_resume: Path to resume checkpoints.
        num_epochs: Number of training epochs.
        fp16: Whether to use mixed precision training.
        gradient_accumulation_steps: Number of steps for gradient accumulation.
    """
    try:
        # DeepSpeed configuration
        ds_config = {
            "train_batch_size": train_loader.batch_size * gradient_accumulation_steps,
            "fp16": {"enabled": fp16},
            "zero_optimization": {"stage": 2},
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-5,
                    "warmup_num_steps": 100
                }
            }
        }

        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        # Resume from checkpoint if available
        if os.path.exists(checkpoint_resume):
            _, client_state = model_engine.load_checkpoint(checkpoint_resume)
            start_epoch = client_state['epoch']
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            train_epoch(model_engine, train_loader, epoch, gradient_accumulation_steps)
            evaluate(model_engine, test_loader, epoch)

            # Save checkpoint
            client_state = {'epoch': epoch + 1}
            model_engine.save_checkpoint(checkpoint_resume, client_state=client_state)

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        raise

def train_epoch(
    model_engine: deepspeed.DeepSpeedEngine,
    train_loader: DataLoader,
    epoch: int,
    gradient_accumulation_steps: int
) -> None:
    """
    Train the model for one epoch.

    Args:
        model_engine: DeepSpeed model engine.
        train_loader: DataLoader for training data.
        epoch: Current epoch number.
        gradient_accumulation_steps: Number of gradient accumulation steps.
    """
    model_engine.train()
    total_loss = 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            try:
                input_ids = batch['input_ids'].to(model_engine.device)
                attention_mask = batch['attention_mask'].to(model_engine.device)
                labels = batch['labels'].to(model_engine.device)

                outputs = model_engine(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

                model_engine.backward(loss)

                if (step + 1) % gradient_accumulation_steps == 0:
                    model_engine.step()
                    model_engine.zero_grad()

                total_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

            except Exception as e:
                logger.error(f"Error during batch processing: {str(e)}")
                continue

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

def evaluate(
    model_engine: deepspeed.DeepSpeedEngine,
    test_loader: DataLoader,
    epoch: int
) -> None:
    """
    Evaluate the model on the test set.

    Args:
        model_engine: DeepSpeed model engine.
        test_loader: DataLoader for testing data.
        epoch: Current epoch number.
    """
    model_engine.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                input_ids = batch['input_ids'].to(model_engine.device)
                attention_mask = batch['attention_mask'].to(model_engine.device)
                labels = batch['labels'].to(model_engine.device)

                outputs = model_engine(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.numel()

            except Exception as e:
                logger.error(f"Error during evaluation batch processing: {str(e)}")
                continue

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions

    logger.info(f"Epoch {epoch + 1} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    
    import os
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerationModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def train_and_evaluate(
    model: TextGenerationModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    checkpoint_resume: str,
    num_epochs: int,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    output_dir: str = "output"
) -> None:
    """
    Train and evaluate the model using DeepSpeed.

    Args:
        model: The text generation model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        optimizer: Configured optimizer.
        scheduler: Learning rate scheduler.
        checkpoint_resume: Path to resume checkpoints.
        num_epochs: Number of training epochs.
        fp16: Whether to use mixed precision training.
        gradient_accumulation_steps: Number of steps for gradient accumulation.
        output_dir: Directory to save model checkpoints and logs.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        ds_config = {
            "train_batch_size": train_loader.batch_size * gradient_accumulation_steps,
            "fp16": {"enabled": fp16},
            "zero_optimization": {"stage": 2},
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-5,
                    "warmup_num_steps": 100
                }
            }
        }

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        start_epoch = 0
        if os.path.exists(checkpoint_resume):
            _, client_state = model_engine.load_checkpoint(checkpoint_resume)
            start_epoch = client_state['epoch']
            logger.info(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            train_loss = train_epoch(model_engine, train_loader, optimizer, scheduler, epoch)
            eval_loss, accuracy = evaluate(model_engine, test_loader, epoch)

            # Save checkpoint
            client_state = {'epoch': epoch + 1}
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}")
            model_engine.save_checkpoint(checkpoint_path, client_state=client_state)

            # Log metrics
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                        f"Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        raise

def train_epoch(
    model_engine: deepspeed.DeepSpeedEngine,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch: int
) -> float:
    """
    Train the model for one epoch.

    Args:
        model_engine: DeepSpeed model engine.
        train_loader: DataLoader for training data.
        optimizer: Configured optimizer.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.

    Returns:
        Average loss for the epoch.
    """
    model_engine.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
        for batch in tepoch:
            try:
                input_ids = batch['input_ids'].to(model_engine.device)
                attention_mask = batch['attention_mask'].to(model_engine.device)
                labels = batch['labels'].to(model_engine.device)

                outputs = model_engine(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

                model_engine.backward(loss)
                model_engine.step()
                total_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                continue

    avg_loss = total_loss / num_batches
    scheduler.step()
    return avg_loss

def evaluate(
    model_engine: deepspeed.DeepSpeedEngine,
    test_loader: DataLoader,
    epoch: int
) -> tuple[float, float]:
    """
    Evaluate the model on the test set.

    Args:
        model_engine: DeepSpeed model engine.
        test_loader: DataLoader for testing data.
        epoch: Current epoch number.

    Returns:
        Tuple of (average loss, accuracy) for the epoch.
    """
    model_engine.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                input_ids = batch['input_ids'].to(model_engine.device)
                attention_mask = batch['attention_mask'].to(model_engine.device)
                labels = batch['labels'].to(model_engine.device)

                outputs = model_engine(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.numel()

            except Exception as e:
                logger.error(f"Error in evaluation batch processing: {str(e)}")
                continue

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

if __name__ == "__main__":
    # Example usage
    model = TextGener