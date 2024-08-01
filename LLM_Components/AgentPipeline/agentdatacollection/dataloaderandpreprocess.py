   
import os
from typing import Union, List, Dict, Any, Optional
import json
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any
from langchain.docstore.document import Document
from datasets import Dataset, DatasetDict, load_dataset

from evaluate import load

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/agent_dataprocessing.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
class AgentDataset:
    """A class to load and process various dataset formats."""

    def __init__(self, source: str):
        """
        Initialize the SampleDataset.

        Args:
            source: Path to the folder containing data files.
        """
        self.source = Path(source)
        self.dataset: Optional[DatasetDict] = None

    def load_data(self) -> DatasetDict:
        """
        Load the dataset from the provided folder.

        Returns:
            Loaded dataset.

        Raises:
            ValueError: If the source is not a directory.
        """
        if not self.source.is_dir():
            raise ValueError("Source must be a directory.")
        
        self.dataset = self._load_from_folder()
        return self.dataset

    def _load_from_folder(self) -> DatasetDict:
        """
        Load datasets from a folder containing various file formats.

        Returns:
            Loaded datasets.
        """
        datasets = {}
        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(self._load_file, file): file
                for file in self.source.iterdir() if file.is_file()
            }
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        file_name, dataset = result
                        datasets[file_name] = dataset
                except Exception as exc:
                    logger.error(f"Error loading file {file}: {exc}")

        return DatasetDict(datasets)

    def _load_file(self, file_path: Path) -> Optional[tuple[str, Dataset]]:
        """
        Load a single file based on its extension.

        Args:
            file_path: Path to the file.

        Returns:
            A tuple containing the file name and loaded dataset, or None if unsupported.
        """
        file_name = file_path.stem
        file_ext = file_path.suffix.lower()

        if file_ext == '.json':
            return file_name, self._load_json(file_path)
        elif file_ext == '.jsonl':
            return file_name, self._load_jsonl(file_path)
        elif file_ext == '.csv':
            return file_name, self._load_csv(file_path)
        elif file_ext in ['.txt', '.md']:
            return file_name, self._load_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return None

    def _load_json(self, file_path: Path) -> Dataset:
        """
        Load JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Loaded dataset.
        """
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_dict(self._flatten_dict(data))

    def _load_jsonl(self, file_path: Path) -> Dataset:
        """
        Load JSONL file.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            Loaded dataset.
        """
        data = []
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return Dataset.from_list(data)

    def _load_csv(self, file_path: Path) -> Dataset:
        """
        Load CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Loaded dataset.
        """
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)

    def _load_text(self, file_path: Path) -> Dataset:
        """
        Load text file (TXT or MD).

        Args:
            file_path: Path to the text file.

        Returns:
            Loaded dataset.
        """
        with file_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        return Dataset.from_dict({'text': lines})

    def _flatten_dict(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Flatten nested dictionaries and lists.

        Args:
            data: Input data structure.

        Returns:
            Flattened dictionary.

        Raises:
            ValueError: If the data structure is invalid.
        """
        if isinstance(data, list):
            return {k: [d.get(k) for d in data] for k in set().union(*data)}
        elif isinstance(data, dict):
            flattened = {}
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    nested = self._flatten_dict(v)
                    flattened.update({f"{k}.{nk}": nv for nk, nv in nested.items()})
                else:
                    flattened[k] = [v]
            return flattened
        else:
            raise ValueError("Invalid data structure. Expected list or dict.")


class AgentDatasetLoader:
    def __init__(self, dataset: DatasetDict):
        """
        Initialize the AdvancedDatasetLoader.

        Args:
            dataset (DatasetDict): The loaded dataset.
        """
        self.dataset = dataset
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, dataset_name: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Load a specific dataset.

        Args:
            dataset_name (str): The name of the dataset to load.

        Returns:
            Tuple[List[str], List(Dict[str, str]]): A tuple containing page content and metadata.
        """
        try:
            dataset_info = self.dataset[dataset_name]
            features = dataset_info.features
            num_rows = len(dataset_info)

            # Generate page content
            page_content = []
            meta_data =[]
            for i in range(max(5, num_rows)):  # Show first 5 rows as an example
                row_content = ", ".join([f"{feature}: {dataset_info[i][feature]}" for feature in features])
                page_content.append(row_content)
                

                # Generate metadata
                metadata = {
                'dataset_name': dataset_name,
                'features': ", ".join(features),
                'row_num': str(i),
                "row_content": row_content,
                }
                meta_data.append(metadata)

            return page_content, meta_data
        except Exception as e:
            self.logger.error(f"Error loading dataset '{dataset_name}': {str(e)}")
            return [], [{}]
    def load_multiple_datasets(self, dataset_names: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
       """
       Load multiple datasets concurrently.
   
       Args:
           dataset_names (List[str]): List of dataset names to load.
   
       Returns:
           Tuple[List[str], List[Dict[str, str]]]: A tuple containing all page contents and metadata.
       """
       all_page_contents = []
       all_metadatas = []
       
       with ThreadPoolExecutor() as executor:
           future_to_dataset = {executor.submit(self.load_dataset, name): name for name in dataset_names}
           for future in as_completed(future_to_dataset):
               dataset_name = future_to_dataset[future]
               try:
                   page_contents, metadatas = future.result()
                   all_page_contents.extend(page_contents)
                   all_metadatas.extend(metadatas)
               except Exception as e:
                   self.logger.error(f"Error processing dataset '{dataset_name}': {str(e)}")
       
       return all_page_contents, all_metadatas
    # def load_multiple_datasets(self, dataset_names: List[str]) -> List[Document]:
    #     """
    #     Load multiple datasets concurrently.

    #     Args:
    #         dataset_names (List[str]): List of dataset names to load.

    #     Returns:
    #         List[Document]: List of loaded documents.
    #     """
    #     documents = []
    #     with ThreadPoolExecutor() as executor:
    #         future_to_dataset = {executor.submit(self.load_dataset, name): name for name in dataset_names}
    #         for future in as_completed(future_to_dataset):
    #             dataset_name = future_to_dataset[future]
    #             try:
    #                 page_content, metadata = future.result()
    #                 if page_content and metadata:
    #                     merged_metadata: Dict[str, str] = {k: v for d in metadata for k, v in d.items()}
    #                     documents.append(Document(page_content="\n".join(page_content), metadata=merged_metadata))
    #             except Exception as e:
    #                 self.logger.error(f"Error processing dataset '{dataset_name}': {str(e)}")
    #     return documents

    def get_dataset_summary(self) -> Dict[str, str]:
        """
        Get a summary of the entire dataset structure.

        Returns:
            Dict[str, str]: Summary of the dataset structure.
        """
        total_datasets = len(self.dataset)
        total_rows = sum(len(dataset) for dataset in self.dataset.values())
        all_features = set()
        for dataset in self.dataset.values():
            all_features.update(dataset.features)

        return {
            'total_datasets': str(total_datasets),
            'total_rows': str(total_rows),
            'all_features': ", ".join(all_features)
        }