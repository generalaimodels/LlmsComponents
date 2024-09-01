from typing import Dict, List
from datasets import Dataset
from tqdm import tqdm


class ConcatDatasetBatch(Dataset):
    """A dataset class for concatenating and batching samples to fixed chunk sizes."""

    def __init__(self, dataset: Dataset, chunk_size: int = 1024, batch_size: int = 32):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.samples = self._preprocess_dataset()

    def _preprocess_dataset(self) -> List[Dict[str, List]]:
        samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "label_g": [],
        }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            for key in buffer:
                buffer[key].extend(sample[key])

            while len(buffer["input_ids"]) >= self.chunk_size * self.batch_size:
                samples.append(self._extract_chunk(buffer))
                buffer = {k: v[self.chunk_size * self.batch_size:] for k, v in buffer.items()}

        if any(len(v) > 0 for v in buffer.values()):
            samples.append(self._pad_and_chunk(buffer))

        return samples

    def _extract_chunk(self, buffer: Dict[str, List]) -> Dict[str, List]:
        chunk = {k: [v[i:i+self.chunk_size] for i in range(0, len(v), self.chunk_size)][:self.batch_size] 
                 for k, v in buffer.items()}
        return chunk

    def _pad_and_chunk(self, buffer: Dict[str, List]) -> Dict[str, List]:
        # Pad to ensure we can create chunks of chunk_size
        for key, value in buffer.items():
            shortfall = self.chunk_size * self.batch_size - len(value)
            if shortfall > 0:
                buffer[key].extend([0 if key != 'labels' else -100] * shortfall)
        return self._extract_chunk(buffer)

    def __getitem__(self, idx: int) -> Dict[str, List]:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)

