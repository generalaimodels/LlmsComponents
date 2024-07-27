from typing import Dict, List, Tuple, Any
from langchain.docstore.document import Document
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, DatasetDict, load_dataset
from dataagent import SampleDataset123
class AdvancedDatasetLoader:
    def __init__(self, dataset: DatasetDict):
        """
        Initialize the AdvancedDatasetLoader.

        Args:
            dataset (DatasetDict): The loaded dataset.
        """
        self.dataset = dataset
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
            dataset_info = self.dataset[dataset_name]
            features = dataset_info.features
            num_rows = len(dataset_info)

            # Generate page content
            page_content = []
            for i in range(min(5, num_rows)):  # Show first 5 rows as an example
                row_content = [f"{feature}: {dataset_info[i][feature]}" for feature in features]
                page_content.extend(row_content)
                page_content.append("---")  # Separator between rows

            # Generate metadata
            metadata = {
                'dataset_name': dataset_name,
                'features': ", ".join(features),
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

# Example usage
if __name__ == "__main__":
    # Sample dataset structure (you would typically load this from a file or API)
    from datasets import DatasetDict, Dataset, Features, Value

    # sample_dataset = DatasetDict({
    #     'prompts': Dataset.from_dict({
    #         'act': ['act1', 'act2'],
    #         'prompt': ['prompt1', 'prompt2']
    #     }, Features({'act': Value('string'), 'prompt': Value('string')})),
    #     'read': Dataset.from_dict({
    #         'text': ['text1', 'text2']
    #     }, Features({'text': Value('string')})),
    #     'testing': Dataset.from_dict({
    #         'prompt': ['prompt1', 'prompt2'],
    #         'expected': ['expected1', 'expected2'],
    #         'generated': ['generated1', 'generated2'],
    #         'accuracy': [0.9, 0.8],
    #         'bleu': [0.7, 0.6],
    #         'chrf': [0.5, 0.4]
    #     }, Features({
    #         'prompt': Value('string'),
    #         'expected': Value('string'),
    #         'generated': Value('string'),
    #         'accuracy': Value('float'),
    #         'bleu': Value('float'),
    #         'chrf': Value('float')
    #     }))
    # })
    dataset=SampleDataset123(source=r"C:\Users\heman\Desktop\components\output").load_data()
    print(dataset)
    loader = AdvancedDatasetLoader(dataset)

    # Load a single dataset
    page_content, metadata = loader.load_dataset('prompts')
    print("Single dataset load:")
    print(f"Page content: {page_content[2:3]}")
    print(f"Metadata: {metadata}")

    # # Load multiple datasets
    # documents = loader.load_multiple_datasets(['prompts', 'read', 'testing'])
    # print(f"\nLoaded {len(documents)} documents")
    # for doc in documents:
    #     print(f"Document content:\n{doc.page_content}")
    #     print(f"Document metadata: {doc.metadata}\n")

    # Get dataset summary
    summary = loader.get_dataset_summary()
    print(f"Dataset summary: {summary}")