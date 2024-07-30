import os
import logging
from typing import List, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/agent_file_processor.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

class AdvancedRAGPipeline:
    def __init__(
        self,
        directory_path: Union[str, Path],
        file_glob: str = "*",
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
    ):
        self.directory_path = Path(directory_path)
        self.file_glob = file_glob
        self.recursive = recursive
        self.exclude_patterns = exclude_patterns or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.distance_strategy = distance_strategy
        
        self.documents = None
        self.chunks = None
        self.vectorstore = None

    def load_documents(self) -> None:
        """Load documents from the specified directory."""
        try:
            loader = DirectoryLoader(
                str(self.directory_path),
                glob=self.file_glob,
                recursive=self.recursive,
                exclude=self.exclude_patterns
            )
            self.documents = loader.load()
            logger.info(f"Loaded {len(self.documents)} documents.")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def split_documents(self) -> None:
        """Split loaded documents into chunks."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self.chunks = splitter.split_documents(self.documents)
            logger.info(f"Split documents into {len(self.chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

    def create_vectorstore(self) -> None:
        """Create a FAISS vectorstore from document chunks."""
        if not self.chunks:
            raise ValueError("No document chunks available. Call split_documents() first.")
        
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vectorstore = FAISS.from_documents(
                self.chunks,
                embeddings,
                distance_strategy=self.distance_strategy
            )
            logger.info("Created FAISS vectorstore.")
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise

    def save_vectorstore(self, save_path: Union[str, Path]) -> None:
        """Save the vectorstore to disk."""
        if not self.vectorstore:
            raise ValueError("No vectorstore available. Call create_vectorstore() first.")
        
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(save_path))
            logger.info(f"Saved vectorstore to {save_path}")
        except Exception as e:
            logger.error(f"Error saving vectorstore: {e}")
            raise

    def load_vectorstore(self, load_path: Union[str, Path]) -> None:
        """Load a vectorstore from disk."""
        try:
            load_path = Path(load_path)
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vectorstore = FAISS.load_local(str(load_path), embeddings)
            logger.info(f"Loaded vectorstore from {load_path}")
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            raise

    def query(self, query: str, k: int = 5) -> List[str]:
        """Query the vectorstore for similar documents."""
        if not self.vectorstore:
            raise ValueError("No vectorstore available. Create or load a vectorstore first.")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"Error querying vectorstore: {e}")
            raise

def main():
    pipeline = AdvancedRAGPipeline("/path/to/documents")
    pipeline.load_documents()
    pipeline.split_documents()
    pipeline.create_vectorstore()
    pipeline.save_vectorstore("/path/to/save/vectorstore")
    
    # Later, you can load the vectorstore and query it
    # pipeline.load_vectorstore("/path/to/save/vectorstore")
    # results = pipeline.query("Your query here")
    # print(results)

if __name__ == "__main__":
    main()

import os
from typing import List, Optional, Union, Callable, Any, Dict
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.base import Docstore

def load_documents(
    directory: str,
    glob: Optional[str] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    recursive: bool = False
) -> List[str]:
    """Load documents from the specified directory with optional filters."""
    try:
        loader = DirectoryLoader(directory, glob=glob, exclude=exclude, recursive=recursive)
        return loader.load()
    except Exception as e:
        print(f"Error loading documents from directory {directory}: {e}")
        return []

def split_documents(
    documents: List[str],
    separators: Optional[List[str]] = None,
    keep_separator: bool = True,
    is_separator_regex: bool = False,
    **kwargs: Any
) -> List[str]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
    try:
        splitter = RecursiveCharacterTextSplitter(separators=separators, keep_separator=keep_separator,
                                                  is_separator_regex=is_separator_regex, **kwargs)
        return splitter.split(documents)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []

def create_faiss_index(
    documents: List[str],
    embedding_function: Union[Callable[[str], List[float]], HuggingFaceEmbeddings],
    relevance_score_fn: Optional[Callable[[float], float]] = None,
    normalize_L2: bool = False,
    distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE
) -> FAISS:
    """Create a FAISS index from the given documents."""
    try:
        docstore = Docstore()  # Assuming a suitable implementation of Docstore
        embeddings = embedding_function(documents)
        index = FAISS.build(embeddings, docstore, distance_strategy)
        return FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            relevance_score_fn=relevance_score_fn,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy
        )
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise

def main():
    directory = "/path/to/directory"
    documents = load_documents(directory, glob="*.txt", recursive=True)
    
    if not documents:
        print("No documents loaded, exiting.")
        return
    
    chunks = split_documents(documents)
    
    if not chunks:
        print("No chunks created, exiting.")
        return
    
    embedding_function = HuggingFaceEmbeddings()  # Assuming the appropriate instantiation
    try:
        faiss_index = create_faiss_index(chunks, embedding_function)
        print("FAISS index created successfully.")
    except Exception as e:
        print(f"Failed to create FAISS index: {e}")

if __name__ == "__main__":
    main()


import os
from typing import List, Optional, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        directory_path: Union[str, Path],
        file_pattern: str = "*.*",
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ):
        self.directory_path = Path(directory_path)
        self.file_pattern = file_pattern
        self.recursive = recursive
        self.exclude_patterns = exclude_patterns or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.distance_strategy = distance_strategy
        self.vectorstore = None

    def load_documents(self) -> List[str]:
        """Load documents from the specified directory."""
        try:
            loader = DirectoryLoader(
                str(self.directory_path),
                glob=self.file_pattern,
                recursive=self.recursive,
                exclude=self.exclude_patterns,
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def split_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks."""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            splits = splitter.split_documents(documents)
            logger.info(f"Split documents into {len(splits)} chunks")
            return splits
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise

    def create_vectorstore(self, splits: List[str]) -> None:
        """Create a FAISS vectorstore from document splits."""
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vectorstore = FAISS.from_documents(
                splits,
                embeddings,
                distance_strategy=self.distance_strategy,
            )
            logger.info("Created FAISS vectorstore")
        except Exception as e:
            logger.error(f"Error creating vectorstore: {str(e)}")
            raise

    def process_documents(self) -> None:
        """Process documents through the entire pipeline."""
        documents = self.load_documents()
        splits = self.split_documents(documents)
        self.create_vectorstore(splits)

    def query(self, query: str, k: int = 5) -> List[str]:
        """Query the vectorstore for similar documents."""
        if not self.vectorstore:
            raise ValueError("Vectorstore has not been created. Run process_documents() first.")
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"Error querying vectorstore: {str(e)}")
            raise

def main():
    try:
        pipeline = RAGPipeline(
            directory_path="/path/to/documents",
            file_pattern="*.txt",
            recursive=True,
            exclude_patterns=["*.log", "*.tmp"],
            chunk_size=500,
            chunk_overlap=50,
        )
        pipeline.process_documents()

        # Example query
        query = "What is the capital of France?"
        results = pipeline.query(query)
        for i, result in enumerate(results, 1):
            print(f"Result {i}:\n{result}\n")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()



import os
from typing import List, Union, Callable, Dict, Any
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

def load_documents(
    dir_path: str, 
    glob_pattern: str = "*.txt", 
    recursive: bool = True, 
    exclude_patterns: Union[None, str, List[str]] = None
) -> List[str]:
    """
    Load documents from a directory.

    :param dir_path: Path to the directory.
    :param glob_pattern: Glob pattern to match files.
    :param recursive: Whether to load files recursively.
    :param exclude_patterns: Patterns to exclude files.
    :return: List of loaded documents.
    """
    try:
        loader = DirectoryLoader(
            dir_path,
            glob=glob_pattern,
            recursive=recursive,
            exclude=exclude_patterns
        )
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def split_documents(
    documents: List[str],
    separators: Union[None, List[str]] = None,
    keep_separator: bool = True
) -> List[str]:
    """
    Split documents into chunks.

    :param documents: List of documents to split.
    :param separators: List of separators for splitting.
    :param keep_separator: Whether to keep the separator in the chunks.
    :return: List of document chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            keep_separator=keep_separator
        )
        chunks = [chunk for doc in documents for chunk in text_splitter.split(doc)]
        return chunks
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []

def create_vector_store(
    chunks: List[str],
    embedding_function: Callable[[str], List[float]],
    distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE
) -> FAISS:
    """
    Create a FAISS vector store with embedded document chunks.

    :param chunks: List of document chunks.
    :param embedding_function: Function to embed the document chunks.
    :param distance_strategy: Strategy to calculate the distance between vectors.
    :return: FAISS vector store.
    """
    try:
        embeddings = [embedding_function(chunk) for chunk in chunks]
        vector_store = FAISS.from_embeddings(
            embeddings=embeddings,
            distance_strategy=distance_strategy
        )
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def main():
    directory_path = "/path/to/directory"
    documents = load_documents(directory_path)
    if not documents:
        print("No documents loaded.")
        return

    chunks = split_documents(documents)
    embedding_function = HuggingFaceEmbeddings.from_pretrained("distilbert-base-uncased").embed
    vector_store = create_vector_store(chunks, embedding_function)

    if vector_store is None:
        print("Failed to create vector store.")
        return

    print("Vector store created successfully.")

if __name__ == '__main__':
    main()