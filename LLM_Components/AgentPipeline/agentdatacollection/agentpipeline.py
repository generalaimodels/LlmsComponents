import os
import logging
from typing import List, Optional, Union,Tuple
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/Agent_file_loading.log',
    filemode='w'
)
logger = logging.getLogger(__name__)



class AgentRAGPipeline:
    def __init__(
        self,
        embedding_model: HuggingFaceEmbeddings,
        directory_path: Union[str, Path],
        file_glob:Union[List[str], Tuple[str], str] = "**/[!.]*",
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
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
                exclude=self.exclude_patterns,
                show_progress= True,
                use_multithreading = True,
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
            
            self.vectorstore = FAISS.from_documents(
                self.chunks,
                self.embedding_model,
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
            self.vectorstore = FAISS.load_local(str(load_path), self.embedding_model)
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
