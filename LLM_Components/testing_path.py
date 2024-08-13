import os
import time
from typing import List, Union, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document


class AdvancedDirectoryLoader:
    """An advanced directory loader with enhanced metadata and processing capabilities."""

    def __init__(
        self,
        path: Union[str, Path],
        glob: Union[List[str], str] = "**/[!.]*",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = True,
        use_multithreading: bool = True,
        max_concurrency: int = 4,
        exclude: Union[List[str], str] = (),
        sample_size: int = 0,
        randomize_sample: bool = False,
        sample_seed: Optional[int] = None,
    ):
        self.path = Path(path)
        self.glob = glob
        self.silent_errors = silent_errors
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.exclude = exclude
        self.sample_size = sample_size
        self.randomize_sample = randomize_sample
        self.sample_seed = sample_seed

    def load(self) -> List[Document]:
        """Load documents from the specified directory with enhanced metadata."""
        loader = DirectoryLoader(
            str(self.path),
            glob=self.glob,
            silent_errors=self.silent_errors,
            load_hidden=self.load_hidden,
            recursive=self.recursive,
            use_multithreading=self.use_multithreading,
            max_concurrency=self.max_concurrency,
            exclude=self.exclude,
            sample_size=self.sample_size,
            randomize_sample=self.randomize_sample,
            sample_seed=self.sample_seed,
        )

        documents = loader.load()
        return self._enhance_documents(documents)

    def _enhance_documents(self, documents: List[Document]) -> List[Document]:
        """Enhance documents with additional metadata."""
        if self.use_multithreading:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                futures = [executor.submit(self._process_document, doc) for doc in documents]
                return [future.result() for future in as_completed(futures)]
        else:
            return [self._process_document(doc) for doc in documents]

    def _process_document(self, document: Document) -> Document:
        """Process a single document to add enhanced metadata."""
        file_path = Path(document.metadata.get("source", ""))
        if not file_path.is_file():
            return document

        start_time = time.time()
        enhanced_metadata = self._get_enhanced_metadata(file_path)
        processing_time = time.time() - start_time

        enhanced_metadata.update({
            "processing_time": processing_time,
            "loader_name": self.__class__.__name__,
        })

        document.metadata.update(enhanced_metadata)
        return document

    def _get_enhanced_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get enhanced metadata for a file."""
        stat = file_path.stat()
        return {
            "file_path": str(file_path.resolve()),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size": stat.st_size,
            "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "parent_directory": str(file_path.parent),
            "is_symlink": file_path.is_symlink(),
            "mime_type": self._get_mime_type(file_path),
            "relative_path": str(file_path.relative_to(self.path)),
        }

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Get MIME type of a file."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"


# Example usage
if __name__ == "__main__":
    loader = AdvancedDirectoryLoader("C:/Users/heman/Desktop/Coding/LlmsComponents/LLM_Components/AgentPipeline/agentcallingpipline")
    documents = loader.load()
    for doc in documents:
        print(f"Document: {doc.page_content[:50]}...")
        print(f"Metadata: {doc.metadata}")
        print("---")