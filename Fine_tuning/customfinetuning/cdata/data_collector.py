import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from itertools import islice

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=1024):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "label_g":[]
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class ConcatDataset_batch(Dataset):
    def __init__(self, dataset, chunk_size=1024, batch_size=32):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k, v in buffer.items()}

            while len(next(iter(buffer.values()))) >= self.chunk_size * self.batch_size:
                self.samples.append({
                    k: [v[i:i + self.chunk_size] for i in range(0, len(v), self.chunk_size)][:self.batch_size] 
                    for k, v in buffer.items()
                })
                buffer = {k: v[self.chunk_size * self.batch_size:] for k, v in buffer.items()}

        # Handle any remaining samples in the buffer
        if any(len(v) > 0 for v in buffer.values()):
            self.samples.append({
                k: [v[i:i + self.chunk_size] for i in range(0, len(v), self.chunk_size)][:self.batch_size]
                for k, v in buffer.items()
            })

    def __getitem__(self, idx):
        if idx is not None:
           return self.samples[idx]
        else:
            raise TypeError(f"Index must be an integer, not {type(idx).__name__}")

    def __len__(self):
        return len(self.samples)
