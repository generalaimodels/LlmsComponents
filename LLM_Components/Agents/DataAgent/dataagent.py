import os
from typing import Union, List, Dict, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict
import json
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleDataset:
    """A class to load and process various dataset formats."""

    def __init__(self, source: Union[str, Dict[str, str]]):
        """
        Initialize the AdvancedDataset.

        Args:
            source: Dataset name or path to the folder containing data files.
        """
        self.source = source
        self.dataset: Optional[Union[Dataset, DatasetDict]] = None

    def load_data(self) -> Union[Dataset, DatasetDict]:
        """
        Load the dataset based on the provided source.

        Returns:
            Loaded dataset.

        Raises:
            ValueError: If the source type is invalid.
        """
        if isinstance(self.source, str):
            if os.path.isdir(self.source):
                self.dataset = self._load_from_folder()
            else:
                self.dataset = load_dataset(self.source,cache_dir='./data')
        elif isinstance(self.source, dict):
            self.dataset = load_dataset(**self.source,cache_dir='./data')
        else:
            raise ValueError("Invalid source type. Expected str or dict.")

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
                executor.submit(self._load_file, os.path.join(self.source, file)): file
                for file in os.listdir(self.source)
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

    def _load_file(self, file_path: str) -> Optional[tuple[str, Dataset]]:
        """
        Load a single file based on its extension.

        Args:
            file_path: Path to the file.

        Returns:
            A tuple containing the file name and loaded dataset, or None if unsupported.
        """
        file_name, file_ext = os.path.splitext(os.path.basename(file_path))
        file_ext = file_ext.lower()

        if file_ext in ['.json', '.jsonl']:
            return file_name, self._load_json(file_path)
        elif file_ext == '.csv':
            return file_name, self._load_csv(file_path)
        elif file_ext in ['.txt', '.md']:
            return file_name, self._load_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return None

    def _load_json(self, file_path: str) -> Dataset:
        """
        Load JSON or JSONL file.

        Args:
            file_path: Path to the JSON or JSONL file.

        Returns:
            Loaded dataset.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_dict(self._flatten_dict(data))

    def _load_csv(self, file_path: str) -> Dataset:
        """
        Load CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Loaded dataset.
        """
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)

    def _load_text(self, file_path: str) -> Dataset:
        """
        Load text file (TXT or MD).

        Args:
            file_path: Path to the text file.

        Returns:
            Loaded dataset.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
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
        
        





class SampleDataset12:
    """A class to load and process various dataset formats."""

    def __init__(self, source: Union[str, Dict[str, str]]):
        """
        Initialize the AdvancedDataset.

        Args:
            source: Dataset name or path to the folder containing data files.
        """
        self.source = source
        self.dataset: Optional[Union[Dataset, DatasetDict]] = None

    def load_data(self) -> Union[Dataset, DatasetDict]:
        """
        Load the dataset based on the provided source.

        Returns:
            Loaded dataset.

        Raises:
            ValueError: If the source type is invalid.
        """
        if isinstance(self.source, str):
            if Path(self.source).is_dir():
                self.dataset = self._load_from_folder()
            else:
                self.dataset = load_dataset(self.source)
        elif isinstance(self.source, dict):
            self.dataset = load_dataset(**self.source)
        else:
            raise ValueError("Invalid source type. Expected str or dict.")

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
                executor.submit(self._load_file, Path(self.source) / file): file
                for file in os.listdir(self.source)
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

        if file_ext in ('.json', '.jsonl'):
            return file_name, self._load_json(file_path)
        if file_ext == '.csv':
            return file_name, self._load_csv(file_path)
        if file_ext in ('.txt', '.md'):
            return file_name, self._load_text(file_path)

        logger.warning(f"Unsupported file type: {file_path}")
        return None

    def _load_json(self, file_path: Path) -> Dataset:
        """
        Load JSON or JSONL file.

        Args:
            file_path: Path to the JSON or JSONL file.

        Returns:
            Loaded dataset.
        """
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_dict(self._flatten_dict(data))

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
        if isinstance(data, dict):
            flattened = {}
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    nested = self._flatten_dict(v)
                    flattened.update({f"{k}.{nk}": nv for nk, nv in nested.items()})
                else:
                    flattened[k] = [v]
            return flattened

        raise ValueError("Invalid data structure. Expected list or dict.")
    
    
import os
from typing import Union, List, Dict, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict
import json
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleDataset123:
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


from typing import Dict, List, Any, Optional
from langchain.docstore.document import Document
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdvancedDatasetLoader:
    def __init__(self, dataset_structure: Dict[str, Dict[str, Any]]):
        """
        Initialize the AdvancedDatasetLoader.

        Args:
            dataset_structure (Dict[str, Dict[str, Any]]): The structure of the dataset.
        """
        self.dataset_structure = dataset_structure
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, file_name: str) -> Optional[Document]:
        """
        Load a specific dataset file.

        Args:
            file_name (str): The name of the file to load.

        Returns:
            Optional[Document]: The loaded document or None if an error occurs.
        """
        try:
            file_data = self.dataset_structure.get(file_name)
            if not file_data:
                raise ValueError(f"File '{file_name}' not found in the dataset structure.")

            features = file_data.get('features', [])
            num_rows = file_data.get('num_rows', 0)

            # Simulating content generation (replace with actual data loading logic)
            page_content = f"Content of {file_name} with {num_rows} rows"
            metadata = {
                'file_name': file_name,
                'features': features,
                'num_rows': num_rows
            }

            return Document(page_content=page_content, metadata=metadata)
        except Exception as e:
            self.logger.error(f"Error loading file '{file_name}': {str(e)}")
            return None

    def load_multiple_datasets(self, file_names: List[str]) -> List[Document]:
        """
        Load multiple dataset files concurrently.

        Args:
            file_names (List[str]): List of file names to load.

        Returns:
            List[Document]: List of loaded documents.
        """
        documents = []
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.load_dataset, file_name): file_name for file_name in file_names}
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    document = future.result()
                    if document:
                        documents.append(document)
                except Exception as e:
                    self.logger.error(f"Error processing file '{file_name}': {str(e)}")
        return documents

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire dataset.

        Returns:
            Dict[str, Any]: Summary of the dataset.
        """
        total_files = len(self.dataset_structure)
        total_rows = sum(file_data.get('num_rows', 0) for file_data in self.dataset_structure.values())
        unique_features = set()
        for file_data in self.dataset_structure.values():
            unique_features.update(file_data.get('features', []))

        return {
            'total_files': total_files,
            'total_rows': total_rows,
            'unique_features': list(unique_features)
        }
        
from typing import Dict, List, Tuple, Any
from langchain.docstore.document import Document
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdvancedDatasetLoader1234:
    def __init__(self, dataset_structure: Dict[str, Dict[str, Any]]):
        """
        Initialize the AdvancedDatasetLoader.

        Args:
            dataset_structure (Dict[str, Dict[str, Any]]): The structure of the dataset.
        """
        self.dataset_structure = dataset_structure
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, dataset_name: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Load a specific dataset.

        Args:
            dataset_name (str): The name of the dataset to load.

        Returns:
            Tuple[List[str], Dict[str, str]]: A tuple containing page content and metadata.
        """
        try:
            dataset_info = self.dataset_structure.get(dataset_name)
            if not dataset_info:
                raise ValueError(f"Dataset '{dataset_name}' not found in the structure.")

            features = dataset_info.get('features', [])
            num_rows = dataset_info.get('num_rows', 0)

            # Generate page content
            page_content = [f"Feature: {feature}" for feature in features]
            page_content.append(f"Number of rows: {num_rows}")

            # Generate metadata
            metadata = {
                'dataset_name': dataset_name,
                'num_features': str(len(features)),
                'num_rows': str(num_rows)
            }

            return page_content, metadata
        except Exception as e:
            self.logger.error(f"Error loading dataset '{dataset_name}': {str(e)}")
            return [], {}

    def load_multiple_datasets(self, dataset_names: List[str]) -> List[Document]:
        """
        Load multiple datasets concurrently.

        Args:
            dataset_names (List[str]): List of dataset names to load.

        Returns:
            List[Document]: List of loaded documents.
        """
        documents = []
        with ThreadPoolExecutor() as executor:
            future_to_dataset = {executor.submit(self.load_dataset, name): name for name in dataset_names}
            for future in as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                try:
                    page_content, metadata = future.result()
                    if page_content and metadata:
                        documents.append(Document(page_content="\n".join(page_content), metadata=metadata))
                except Exception as e:
                    self.logger.error(f"Error processing dataset '{dataset_name}': {str(e)}")
        return documents

    def get_dataset_summary(self) -> Dict[str, str]:
        """
        Get a summary of the entire dataset structure.

        Returns:
            Dict[str, str]: Summary of the dataset structure.
        """
        total_datasets = len(self.dataset_structure)
        total_rows = sum(dataset_info.get('num_rows', 0) for dataset_info in self.dataset_structure.values())
        unique_features = set()
        for dataset_info in self.dataset_structure.values():
            unique_features.update(dataset_info.get('features', []))

        return {
            'total_datasets': str(total_datasets),
            'total_rows': str(total_rows),
            'unique_features': ", ".join(unique_features)
        }

# # Example usage
# if __name__ == "__main__":
#     # # Sample dataset structure
#     # dataset_structure = {
#     #     'prompts': {'features': ['act', 'prompt'], 'num_rows': 153},
#     #     'read': {'features': ['text'], 'num_rows': 26},
#     #     'testing': {'features': ['prompt', 'expected', 'generated', 'accuracy', 'bleu', 'chrf'], 'num_rows': 128}
#     # }
#     dataset=SampleDataset123(source=r"C:\Users\heman\Desktop\components\output").load_data()
#     print(dataset)

#     # loader = AdvancedDatasetLoader(dataset)

#     loader = AdvancedDatasetLoader(dataset)

#     # Load a single dataset
#     page_content, metadata = loader.load_dataset('prompts')
#     print("Single dataset load:")
#     print(f"Page content: {page_content}")
#     print(f"Metadata: {metadata}")

#     # Load multiple datasets
#     documents = loader.load_multiple_datasets(['prompts', 'read', 'testing'])
#     print(f"\nLoaded {len(documents)} documents")
#     for doc in documents:
#         print(f"Document content:\n{doc.page_content}")
#         print(f"Document metadata: {doc.metadata}\n")

#     # Get dataset summary
#     summary = loader.get_dataset_summary()
#     print(f"Dataset summary: {summary}")


#     # Load a single dataset
# # document = loader.load_dataset('file1.csv')
# # if document:
# #     print(f"Loaded document: {document.page_content}")
# #     print(f"Metadata: {document.metadata}")
# # # Load multiple datasets
# # documents = loader.load_multiple_datasets(['file1.csv', 'file2.csv', 'file3.csv'])
# # print(f"Loaded {len(documents)} documents")
# # # Get dataset summary
# # summary = loader.get_dataset_summary()
# # print(f"Dataset summary: {summary}")