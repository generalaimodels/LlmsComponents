import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Accelerate module with ImageNet and torchvision models")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageNet dataset")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    return parser.parse_args()


def get_dataloaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create and return train and validation dataloaders."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageNet(data_dir, split='train', transform=transform)
    val_dataset = datasets.ImageNet(data_dir, split='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def get_model(model_name: str) -> nn.Module:
    """Create and return the specified torchvision model."""
    try:
        model = getattr(models, model_name)(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1000)  # ImageNet has 1000 classes
        return model
    except AttributeError:
        raise ValueError(f"Model {model_name} not found in torchvision.models")


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler, accelerator: Accelerator) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs, labels = batch
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, accelerator: Accelerator) -> float:
    """Evaluate the model and return the accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


def main():
    args = parse_args()
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)
    model = get_model(args.model)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.num_epochs,
                           steps_per_epoch=len(train_loader))

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, accelerator)
        accuracy = evaluate(model, val_loader, accelerator)

        accelerator.print(f"Epoch {epoch+1}/{args.num_epochs}:")
        accelerator.print(f"  Train Loss: {train_loss:.4f}")
        accelerator.print(f"  Validation Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if accelerator.is_main_process:
                accelerator.save_state(os.path.join(args.output_dir, "best_model"))

    accelerator.print(f"Best Validation Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()

# python script_name.py --data_dir /path/to/imagenet --model resnet50 --batch_size 64 --num_epochs 5 --learning_rate 1e-3 --mixed_precision fp16 --output_dir ./output