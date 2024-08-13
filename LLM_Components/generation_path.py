import os
import mimetypes
from datetime import datetime
from typing import List, Union, Tuple, Dict, Any
from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document


class AdvancedDocumentProcessor:
    def __init__(self, directory_path: str, file_types: Union[List[str], Tuple[str, ...], str] = "**/[!.]*") -> None:
        """
        Initialize the AdvancedDocumentProcessor.

        :param directory_path: Path to the directory containing documents.
        :param file_types: Glob pattern to filter files by type.
        """
        self.directory_loader = DirectoryLoader(
            path=directory_path,
            glob=file_types,
            silent_errors=False,
            load_hidden=False,
            recursive=True,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=4
        )

    def process_documents(self) -> List[Document]:
        """
        Process documents in the specified directory and extract metadata.

        :return: List of Document objects with metadata.
        """
        documents = []
        for file_path in self.directory_loader.load():
            metadata = self.extract_metadata(file_path)
            with open(file_path, 'r') as file:
                content = file.read()
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)
        return documents

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file.

        :param file_path: Path to the file.
        :return: Dictionary of metadata attributes.
        """
        file_stats = os.stat(file_path)
        file_type, _ = mimetypes.guess_type(file_path)

        metadata = {
            'absolute_path': os.path.abspath(file_path),
            'creation_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            'size': file_stats.st_size,
            'file_type': file_type or 'unknown',
            'directory': os.path.dirname(file_path),
            'url': self.generate_file_url(file_path),
            'processing_time': self.estimate_processing_time(file_stats.st_size),
        }
        return metadata

    @staticmethod
    def generate_file_url(file_path: str) -> str:
        """
        Generate a file URL from the file path.

        :param file_path: Path to the file.
        :return: URL string.
        """
        return f"file://{os.path.abspath(file_path)}"

    @staticmethod
    def estimate_processing_time(file_size: int) -> str:
        """
        Estimate the processing time for a file based on its size.

        :param file_size: Size of the file in bytes.
        :return: Estimated processing time as a string.
        """
        # Arbitrary formula to estimate time, could be improved with real benchmarking
        estimate = file_size / (1024 * 1024 * 10)  # assuming 10 MB/s processing capability
        return f"{estimate:.2f} seconds"


