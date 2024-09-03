import os
import time
from typing import List, Union, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import DirectoryLoader
from langchain.docstore.document import Document
from rag_config import DirectoryLoaderConfig

class DirectoryLoader:
    """An advanced directory loader with enhanced metadata and processing capabilities."""

    def __init__(
        self,
       config:DirectoryLoaderConfig
    ):
        self.config=config
       

    def load(self) -> List[Document]:
        """Load documents from the specified directory with enhanced metadata."""
        loader = DirectoryLoader(
            str(Path(self.config.path).resolve()),
            glob=self.config.glob,
            silent_errors=self.config.silent_errors,
            load_hidden=self.config.load_hidden,
            recursive=self.config.recursive,
            use_multithreading=self.config.use_multithreading,
            max_concurrency=self.config.max_concurrency,
            exclude=self.config.exclude,
            sample_size=self.config.sample_size,
            randomize_sample=self.config.randomize_sample,
            sample_seed=self.config.sample_seed,
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
            "source": self._normalize_path(file_path),
            "file_path": f"file://{self._normalize_path(file_path)}",
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size": stat.st_size,
            "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "parent_directory": self._normalize_path(file_path.parent),
            "is_symlink": file_path.is_symlink(),
            "mime_type": self._get_mime_type(file_path),
            "relative_path": self._normalize_path(file_path.relative_to(self.path)),
        }

    @staticmethod
    def _normalize_path(path: Path) -> str:
        """Normalize path to use forward slashes."""
        return str(path).replace(os.sep, '/')

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Get MIME type of a file."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"




