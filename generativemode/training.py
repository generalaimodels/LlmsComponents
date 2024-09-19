
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced PyTorch Distributed Training Script

This script implements a scalable and robust training loop using PyTorch,
focusing on advanced parallelism, distribution, and multiprocessing,
while effectively utilizing hardware resources. It supports saving and
checkpointing best weights, tracking epochs, managing loss, and handling
large models via distributed strategies.

Author: OpenAI Assistant
"""

import argparse
import os
import sys
import time
import logging
import builtins
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training Example')
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size per process (GPU)')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--data-path', default='./data', type=str, help='Path to dataset')
    parser.add_argument('--save-path', default='./models', type=str, help='Path to save models and checkpoints')
    parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=-1, type=int, help='Number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='Node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='URL to set up distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='Distributed backend')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training')
    args = parser.parse_args()
    return args


def setup_logging() -> None:
    """Sets up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MyDataset(Dataset):
    """Custom Dataset for demonstration purposes."""

    def __init__(self, data_path: str, train: bool = True) -> None:
        super().__init__()
        self.data_path = data_path
        self.train = train
        # TODO: Implement data loading logic

    def __len__(self) -> int:
        return 10000  # TODO: Replace with actual dataset size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Replace with actual data retrieval logic
        data = torch.randn(10)
        target = torch.randint(0, 2, (1,)).long().squeeze()
        return data, target


def build_model() -> nn.Module:
    """Builds the neural network model."""
    model = nn.Sequential(
        nn.Linear(10, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    return model


def save_checkpoint(state: Dict[str, Any], is_best: bool, save_path: str) -> None:
    """Saves the model checkpoint."""
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, 'model_best.pth')
        torch.save(state, best_filename)
    logging.info(f"Checkpoint saved to '{filename}'{' (best model)' if is_best else ''}")


def adjust_learning_rate(optimizer: optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    """Adjusts learning rate based on epoch."""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info(f"Adjusted learning rate to {lr}")


def train_one_epoch(train_loader: DataLoader, model: nn.Module, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device, epoch: int,
                    args: argparse.Namespace) -> float:
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_time = time.time()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if i % 100 == 0:
            logging.info(f'Epoch [{epoch}][{i}/{len(train_loader)}] - Loss: {loss.item():.4f}')

    # Aggregate loss across all processes
    total_loss = torch.tensor(running_loss, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    if args.world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    average_loss = total_loss.item() / total_samples_tensor.item()

    elapsed_time = time.time() - start_time
    logging.info(f'Epoch [{epoch}] Training Loss: {average_loss:.4f} - Time: {elapsed_time:.2f}s')
    return average_loss


def validate(val_loader: DataLoader, model: nn.Module, criterion: nn.Module,
             device: torch.device, epoch: int, args: argparse.Namespace) -> float:
    """Performs validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_samples += batch_size

    # Aggregate metrics across all processes
    total_loss = torch.tensor(running_loss, device=device)
    total_correct = torch.tensor(correct, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    if args.world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    average_loss = total_loss.item() / total_samples_tensor.item()
    accuracy = total_correct.item() / total_samples_tensor.item()

    elapsed_time = time.time() - start_time
    logging.info(f'Epoch [{epoch}] Validation Loss: {average_loss:.4f} '
                 f'Accuracy: {accuracy:.4f} - Time: {elapsed_time:.2f}s')
    return average_loss


def main_worker(gpu: int, args: argparse.Namespace) -> None:
    """Main worker function for each process."""
    try:
        args.gpu = gpu
        if args.gpu is not None:
            logging.info(f"Using GPU: {args.gpu} for training")

        if args.multiprocessing_distributed:
            if args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.world_size == -1:
                args.world_size = int(os.environ["WORLD_SIZE"])
            args.rank = args.rank * torch.cuda.device_count() + gpu
            dist.init_process_group(backend=args.backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
            torch.distributed.barrier()
        else:
            dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=1, rank=0)

        # Suppress printing for non-master processes
        if args.rank != 0:
            def print_pass(*args, **kwargs):
                pass
            builtins.print = print_pass

        # Set device and seed
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Build model
        model = build_model()
        model.to(device)

        # Wrap model for distributed training
        if args.multiprocessing_distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        # Define loss function, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # Optionally resume from a checkpoint
        start_epoch = 0
        best_metric = None
        if args.resume:
            if os.path.isfile(args.resume):
                logging.info(f"Loading checkpoint '{args.resume}'")
                loc = f'cuda:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
                start_epoch = checkpoint['epoch']
                best_metric = checkpoint['best_metric']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                logging.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                logging.warning(f"No checkpoint found at '{args.resume}'")

        # Data loading code
        train_dataset = MyDataset(args.data_path, train=True)
        val_dataset = MyDataset(args.data_path, train=False)

        if args.multiprocessing_distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                  sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                sampler=val_sampler, num_workers=4, pin_memory=True)

        # Training Loop
        for epoch in range(start_epoch, args.epochs):
            if args.multiprocessing_distributed:
                train_sampler.set_epoch(epoch)

            adjust_learning_rate(optimizer, epoch, args)

            train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, args)
            val_loss = validate(val_loader, model, criterion, device, epoch, args)

            # Remember best metric and save checkpoint
            is_best = False
            if best_metric is None or val_loss < best_metric:
                best_metric = val_loss
                is_best = True

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.world_size == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_metric': best_metric,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.save_path)

            scheduler.step()

    except Exception as e:
        logging.exception(f"Exception in main worker: {e}")
        sys.exit(1)


def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging()

    if args.gpu is not None:
        logging.warning('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.multiprocessing_distributed:
        args.world_size = torch.cuda.device_count() * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(args,))
    else:
        main_worker(args.gpu, args)


if __name__ == '__main__':
    main()




import argparse
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, DistributedSampler, Dataset

# Define your model, dataset, and any other required components here


class CustomDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Initialize your dataset

    def __len__(self) -> int:
        # Return the total number of samples
        return 10000

    def __getitem__(self, index: int) -> Any:
        # Return a single sample
        return torch.randn(3, 224, 224), random.randint(0, 9)


class CustomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Define your model architecture
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 222 * 222, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    is_best: bool,
    filename: str = "checkpoint.pth.tar",
    best_name: str = "model_best.pth.tar",
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / filename)
    if is_best:
        torch.save(state["state_dict"], checkpoint_dir / best_name)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[_LRScheduler],
    checkpoint_path: Path,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    best_metric = checkpoint["best_metric"]
    return epoch, best_metric


def init_process_group(
    backend: str = "nccl",
    init_method: str = "env://",
    world_size: int = -1,
    rank: int = -1,
) -> None:
    dist.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size, rank=rank
    )


def cleanup() -> None:
    dist.destroy_process_group()


def train(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    # Set up the device
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Initialize the process group
    init_process_group(world_size=world_size, rank=rank)

    # Create the model, optimizer, and scheduler
    model = CustomModel().to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # Prepare the dataset and data loaders
    train_dataset = CustomDataset()
    val_dataset = CustomDataset()

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Variables for tracking best performance
    best_metric = float("inf")
    start_epoch = 0

    # Load checkpoint if resuming
    if args.resume and args.checkpoint_path.exists():
        start_epoch, best_metric = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint_path
        )

    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Reduce running_loss across all processes
        total_loss = torch.tensor(running_loss, device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss /= world_size

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct += (predicted == targets).sum().item()

        # Reduce validation metrics across all processes
        val_loss_tensor = torch.tensor(val_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)

        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        val_loss = val_loss_tensor.item() / world_size
        val_accuracy = correct_tensor.item() / total_samples_tensor.item()

        scheduler.step(val_loss)

        # Only the main process should save checkpoints and print logs
        if rank == 0:
            is_best = val_loss < best_metric
            best_metric = min(val_loss, best_metric)

            checkpoint_state = {
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(
                checkpoint_state,
                args.checkpoint_dir,
                is_best,
            )

            print(
                f"Epoch [{epoch+1}/{args.epochs}], "
                f"Training Loss: {total_loss.item():.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

        # Optional early stopping
        if val_loss > best_metric + args.early_stop_threshold:
            break

    cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed PyTorch Training")
    parser.add_argument("--epochs", type=int, default=100, help="number of total epochs")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="mini-batch size per process"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="number of data loader workers"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("./checkpoints/checkpoint.pth.tar"),
        help="path to the checkpoint file",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=0.01,
        help="early stopping threshold",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()