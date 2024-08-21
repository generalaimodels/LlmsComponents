from typing import Dict, Any, Optional, List
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedDataLoader:
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloaders: Dict[str, DataLoader] = {}

    def validate_dataset(self) -> None:
        if not isinstance(self.dataset, dict):
            raise ValueError("Dataset must be a dictionary")
        
        for split, data in self.dataset.items():
            if not isinstance(data, dict):
                raise ValueError(f"Data for split '{split}' must be a dictionary")
            
            if 'features' not in data or 'rows_num' not in data:
                raise ValueError(f"Split '{split}' is missing 'features' or 'rows_num'")
            
            if not isinstance(data['features'], list) or not all(isinstance(f, str) for f in data['features']):
                raise ValueError(f"'features' in split '{split}' must be a list of strings")
            
            if not isinstance(data['rows_num'], int) or data['rows_num'] <= 0:
                raise ValueError(f"'rows_num' in split '{split}' must be a positive integer")

    def process_dataset(self, split: str) -> Dataset:
        features = self.dataset[split]['features']
        rows_num = self.dataset[split]['rows_num']
        
        # Here you would implement your actual data processing logic
        # For now, we'll just create a dummy dataset
        data = {feature: torch.rand(rows_num) for feature in features}
        return Dataset.from_dict(data)

    def create_dataloaders(self) -> None:
        try:
            self.validate_dataset()
        except ValueError as e:
            logging.error(f"Dataset validation failed: {str(e)}")
            return

        for split, data in self.dataset.items():
            try:
                processed_dataset = self.process_dataset(split)
                self.dataloaders[split] = DataLoader(
                    processed_dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle if split == 'train' else False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory
                )
                logging.info(f"Successfully created DataLoader for '{split}' split")
            except Exception as e:
                logging.error(f"Error creating DataLoader for '{split}' split: {str(e)}")

    def get_dataloader(self, split: str) -> Optional[DataLoader]:
        return self.dataloaders.get(split)

    def get_available_splits(self) -> List[str]:
        return list(self.dataloaders.keys())

def load_llm_dataset(path: str) -> Dict[str, Any]:
    try:
        dataset = load_dataset(path)
        return {split: {"features": list(data.features.keys()), "rows_num": len(data)} 
                for split, data in dataset.items()}
    except Exception as e:
        logging.error(f"Error loading dataset from {path}: {str(e)}")
        return {}

def main() -> None:
    path = "path/to/your/dataset"
    llm_dataset = load_llm_dataset(path)

    if not llm_dataset:
        logging.error("Failed to load dataset. Exiting.")
        return

    data_loader = AdvancedDataLoader(llm_dataset)
    data_loader.create_dataloaders()

    available_splits = data_loader.get_available_splits()
    for split in available_splits:
        loader = data_loader.get_dataloader(split)
        if loader:
            logging.info(f"Processing data for '{split}' split")
            for batch in loader:
                # Process your batch here
                pass
        else:
            logging.warning(f"No DataLoader available for '{split}' split")

if __name__ == "__main__":
    main()