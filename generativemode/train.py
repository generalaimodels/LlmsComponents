
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""An example of an efficient, robust, and distributed training loop in PyTorch."""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load_state, save_state
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Distributed PyTorch Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend.')
    parser.add_argument('--world-size', type=int, default=-1, help='Number of processes participating in the job.')
    parser.add_argument('--rank', type=int, default=-1, help='The rank of this process.')
    parser.add_argument('--dist-url', type=str, default='env://', help='URL used to set up distributed training.')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save checkpoints.')
    return parser.parse_args()

class DummyDataset(Dataset):
    """A dummy dataset for example purposes."""
    def __len__(self) -> int:
        return 10000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(10)
        y = torch.randint(0, 2, (1,))
        return x, y

def setup_logging(rank: int) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] Rank {rank} %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def setup_distributed(args: argparse.Namespace) -> None:
    """Initialize the distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        logging.warning('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        return

    args.distributed = True
    dist.init_process_group(
        backend=args.backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()
    torch.cuda.set_device(args.local_rank)

def cleanup() -> None:
    """Cleanup distributed environment."""
    dist.destroy_process_group()

def main() -> None:
    args = parse_args()
    setup_distributed(args)
    setup_logging(args.rank)

    try:
        train(args)
    except Exception as e:
        logging.error(f'An exception occurred: {e}', exc_info=True)
        cleanup()
        sys.exit(1)
    else:
        cleanup()

@record
def train(args: argparse.Namespace) -> None:
    """Train the model."""
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = nn.Linear(10, 2).to(device)

    # Wrap model for distributed training
    if args.distributed:
        model = FSDP(model)

    # Create optimizer
    if args.distributed:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=optim.SGD,
            lr=args.lr
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # Create dataset and dataloader
    dataset = DummyDataset()
    if args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )

    # Optionally, load checkpoint
    start_epoch = load_checkpoint(model, optimizer, args)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.distributed and sampler is not None:
            sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = nn.functional.cross_entropy(output, target.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                logging.info(
                    f"Epoch [{epoch}/{args.epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()
        # Save checkpoint
        if not args.distributed or args.rank == 0:
            save_checkpoint(model, optimizer, epoch, args)

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    args: argparse.Namespace
) -> None:
    """Save model and optimizer state using torch.distributed.checkpoint."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.save_dir, f'checkpoint_{epoch}')
    state: Dict[str, Any] = {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
    }
    writer = FileSystemWriter(checkpoint_dir)
    save_state(state, writer)
    logging.info(f'Saved checkpoint at {checkpoint_dir}')

def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace
) -> int:
    """Load model and optimizer state using torch.distributed.checkpoint."""
    latest_checkpoint_dir = find_latest_checkpoint_dir(args)
    if latest_checkpoint_dir is None:
        return 0
    state: Dict[str, Any] = {
        'epoch': None,
        'model': model,
        'optimizer': optimizer,
    }
    reader = FileSystemReader(latest_checkpoint_dir)
    load_state(state, reader)
    epoch: Optional[int] = state['epoch']
    logging.info(f'Loaded checkpoint from {latest_checkpoint_dir}')
    return epoch + 1 if epoch is not None else 0

def find_latest_checkpoint_dir(args: argparse.Namespace) -> Optional[str]:
    """Find the latest checkpoint directory."""
    if not os.path.isdir(args.save_dir):
        return None
    dirs = os.listdir(args.save_dir)
    checkpoint_dirs = [
        d for d in dirs
        if d.startswith('checkpoint_') and os.path.isdir(os.path.join(args.save_dir, d))
    ]
    if not checkpoint_dirs:
        return None
    latest_checkpoint_dir = max(
        checkpoint_dirs,
        key=lambda x: int(x.split('_')[1])
    )
    return os.path.join(args.save_dir, latest_checkpoint_dir)

if __name__ == '__main__':
    main()