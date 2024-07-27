from typing import List, Dict, Union, Optional, Any, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class AdvancedVectorRetrieval_test:
    def __init__(
        self,
        embedding_model: str = "distilbert-base-uncased",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ) -> None:
        """
        Initialize the AdvancedVectorRetrieval system.

        Args:
            embedding_model (str): The name of the Hugging Face model to use for embeddings.
            distance_strategy (DistanceStrategy): The strategy to use for calculating distances.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.distance_strategy = distance_strategy
        self.documents: List[Document] = []
        self.document_embeddings: List[np.ndarray] = []

    def add_documents(
        self,
        documents: Union[List[str], Dict[str, List[str]], str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the retrieval system.

        Args:
            documents: The documents to add. Can be a list of strings, a dictionary of strings to list of strings,
                       or a single string.
            metadata: Optional metadata for each document.
        """
        if isinstance(documents, str):
            documents = [documents]
        elif isinstance(documents, dict):
            documents = [f"{k}: {' '.join(v)}" for k, v in documents.items()]

        new_docs = [Document(page_content=doc, metadata=meta or {}) 
                    for doc, meta in zip(documents, metadata or [{}] * len(documents))]
        
        self.documents.extend(new_docs)
        
        new_embeddings = self.embeddings.embed_documents([doc.page_content for doc in new_docs])
        self.document_embeddings.extend(new_embeddings)

    def _calculate_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between query and documents based on the chosen distance strategy.

        Args:
            query_embedding (np.ndarray): The embedding of the query.

        Returns:
            np.ndarray: Array of similarity scores.
        """
        if self.distance_strategy == DistanceStrategy.COSINE:
            return cosine_similarity([query_embedding], self.document_embeddings)[0]
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return -np.linalg.norm(np.array(self.document_embeddings) - query_embedding, axis=1)
        else:
            raise ValueError(f"Unsupported distance strategy: {self.distance_strategy}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        depth_search: int = 1,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the most relevant documents for a given query.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return.
            depth_search (int): The depth of the search for re-ranking.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing the retrieved documents and their relevance scores.
        """
        query_embedding = self.embeddings.embed_query(query)
        similarities = self._calculate_similarity(query_embedding)
        
        top_indices = np.argsort(similarities)[-top_k * depth_search:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc_query_similarity = similarities[idx]
            
            # Re-rank based on document-query similarity
            doc_embedding = self.document_embeddings[idx]
            relevance_score = (doc_query_similarity + cosine_similarity([query_embedding], [doc_embedding])[0][0]) / 2
            
            results.append((doc, float(relevance_score)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        depth_search: int = 1,
    ) -> List[List[Tuple[Document, float]]]:
        """
        Perform batch retrieval for multiple queries.

        Args:
            queries (List[str]): List of query strings.
            top_k (int): The number of top results to return for each query.
            depth_search (int): The depth of the search for re-ranking.

        Returns:
            List[List[Tuple[Document, float]]]: List of retrieval results for each query.
        """
        return [self.retrieve(query, top_k, depth_search) for query in tqdm(queries, desc="Processing queries")]

    def update_document(self, index: int, new_content: str, new_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update a document in the retrieval system.

        Args:
            index (int): The index of the document to update.
            new_content (str): The new content of the document.
            new_metadata (Optional[Dict[str, Any]]): The new metadata for the document.
        """
        if 0 <= index < len(self.documents):
            self.documents[index] = Document(page_content=new_content, metadata=new_metadata or {})
            self.document_embeddings[index] = self.embeddings.embed_query(new_content)
        else:
            raise IndexError("Document index out of range")

    def remove_document(self, index: int) -> None:
        """
        Remove a document from the retrieval system.

        Args:
            index (int): The index of the document to remove.
        """
        if 0 <= index < len(self.documents):
            del self.documents[index]
            del self.document_embeddings[index]
        else:
            raise IndexError("Document index out of range")

    def save_state(self, file_path: str) -> None:
        """
        Save the current state of the retrieval system to a file.

        Args:
            file_path (str): The path to the file where the state will be saved.
        """
        state = {
            "documents": [(doc.page_content, doc.metadata) for doc in self.documents],
            "document_embeddings": self.document_embeddings,
            "embedding_model": self.embeddings.model_name,
            "distance_strategy": self.distance_strategy,
        }
        with open(file_path, 'wb') as file:
            np.save(file, state, allow_pickle=True)

    def load_state(self, file_path: str) -> None:
        """
        Load the state of the retrieval system from a file.

        Args:
            file_path (str): The path to the file from which the state will be loaded.
        """
        with open(file_path, 'rb') as file:
            state = np.load(file, allow_pickle=True).item()
        self.documents = [Document(page_content=content, metadata=meta) for content, meta in state["documents"]]
        self.document_embeddings = state["document_embeddings"]
        self.embeddings = HuggingFaceEmbeddings(model_name=state["embedding_model"])
        self.distance_strategy = state["distance_strategy"]

    def get_document(self, index: int) -> Document:
        """
        Get a document from the retrieval system by its index.

        Args:
            index (int): The index of the document to retrieve.

        Returns:
            Document: The document at the specified index.
        """
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            raise IndexError("Document index out of range")
        

from typing import List, Dict, Union, Optional, Any, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class AdvancedVectorRetrieval:
    def __init__(
        self,
        embedding_model: str = "distilbert-base-uncased",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ) -> None:
        """
        Initialize the AdvancedVectorRetrieval system.

        Args:
            embedding_model (str): The name of the Hugging Face model to use for embeddings.
            distance_strategy (DistanceStrategy): The strategy to use for calculating distances.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.distance_strategy = distance_strategy
        self.documents: List[Document] = []
        self.document_embeddings: List[np.ndarray] = []

    def add_documents(
        self,
        documents: Union[List[str], Dict[str, List[str]], str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the retrieval system.

        Args:
            documents: The documents to add. Can be a list of strings, a dictionary of strings to list of strings,
                       or a single string.
            metadata: Optional metadata for each document.
        """
        if isinstance(documents, str):
            documents = [documents]
        elif isinstance(documents, dict):
            documents = [f"{k}: {' '.join(v)}" for k, v in documents.items()]

        new_docs = [Document(page_content=doc, metadata=meta or {}) 
                    for doc, meta in zip(documents, metadata or [{}] * len(documents))]
        
        self.documents.extend(new_docs)
        
        new_embeddings = self.embeddings.embed_documents([doc.page_content for doc in new_docs])
        self.document_embeddings.extend(new_embeddings)

    def _calculate_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between query and documents based on the chosen distance strategy.

        Args:
            query_embedding (np.ndarray): The embedding of the query.

        Returns:
            np.ndarray: Array of similarity scores.
        """
        if self.distance_strategy == DistanceStrategy.COSINE:
            return cosine_similarity([query_embedding], self.document_embeddings)[0]
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return -np.linalg.norm(np.array(self.document_embeddings) - query_embedding, axis=1)
        else:
            raise ValueError(f"Unsupported distance strategy: {self.distance_strategy}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        depth_search: int = 1,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the most relevant documents for a given query.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return.
            depth_search (int): The depth of the search for re-ranking.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing the retrieved documents and their relevance scores.
        """
        query_embedding = self.embeddings.embed_query(query)
        similarities = self._calculate_similarity(query_embedding)
        
        top_indices = np.argsort(similarities)[-top_k * depth_search:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc_query_similarity = similarities[idx]
            
            # Re-rank based on document-query similarity
            doc_embedding = self.document_embeddings[idx]
            relevance_score = (doc_query_similarity + cosine_similarity([query_embedding], [doc_embedding])[0][0]) / 2
            
            results.append((doc, float(relevance_score)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        depth_search: int = 1,
    ) -> List[List[Tuple[Document, float]]]:
        """
        Perform batch retrieval for multiple queries.

        Args:
            queries (List[str]): List of query strings.
            top_k (int): The number of top results to return for each query.
            depth_search (int): The depth of the search for re-ranking.

        Returns:
            List[List[Tuple[Document, float]]]: List of retrieval results for each query.
        """
        return [self.retrieve(query, top_k, depth_search) for query in tqdm(queries, desc="Processing queries")]

    def update_document(self, index: int, new_content: str, new_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update a document in the retrieval system.

        Args:
            index (int): The index of the document to update.
            new_content (str): The new content of the document.
            new_metadata (Optional[Dict[str, Any]]): The new metadata for the document.
        """
        if 0 <= index < len(self.documents):
            self.documents[index] = Document(page_content=new_content, metadata=new_metadata or {})
            self.document_embeddings[index] = self.embeddings.embed_query(new_content)
        else:
            raise IndexError("Document index out of range")

    def remove_document(self, index: int) -> None:
        """
        Remove a document from the retrieval system.

        Args:
            index (int): The index of the document to remove.
        """
        if 0 <= index < len(self.documents):
            del self.documents[index]
            del self.document_embeddings[index]
        else:
            raise IndexError("Document index out of range")

    def save_state(self, file_path: str) -> None:
        """
        Save the current state of the retrieval system to a file.

        Args:
            file_path (str): The path to the file where the state will be saved.
        """
        state = {
            "documents": [(doc.page_content, doc.metadata) for doc in self.documents],
            "document_embeddings": self.document_embeddings,
            "embedding_model": self.embeddings.model_name,
            "distance_strategy": self.distance_strategy,
        }
        with open(file_path, 'wb') as file:
            np.save(file, state, allow_pickle=True)

    def load_state(self, file_path: str) -> None:
        """
        Load the state of the retrieval system from a file.

        Args:
            file_path (str): The path to the file from which the state will be loaded.
        """
        with open(file_path, 'rb') as file:
            state = np.load(file, allow_pickle=True).item()
        self.documents = [Document(page_content=content, metadata=meta) for content, meta in state["documents"]]
        self.document_embeddings = state["document_embeddings"]
        self.embeddings = HuggingFaceEmbeddings(model_name=state["embedding_model"])
        self.distance_strategy = state["distance_strategy"]

    def get_document(self, index: int) -> Document:
        """
        Get a document from the retrieval system by its index.

        Args:
            index (int): The index of the document to retrieve.

        Returns:
            Document: The document at the specified index.
        """
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            raise IndexError("Document index out of range")


# Example usage (continued)
def example_usage():
    retrieval_system = AdvancedVectorRetrieval()

    documents = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
        "The capital of Portugal is Lisbon."
    ]

    # Add documents to the retrieval system
    retrieval_system.add_documents(documents)

    # Retrieve the most relevant document for a query
    query = "What is the capital of France?"
    top_docs = retrieval_system.retrieve(query, top_k=3)
    
    print("Top documents for query:", query)
    for doc, score in top_docs:
        print(f"Document: {doc.page_content}, Score: {score}")

    # Batch retrieval for multiple queries
    queries = [
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]
    batch_results = retrieval_system.batch_retrieve(queries, top_k=2)
    
    for q, results in zip(queries, batch_results):
        print(f"\nTop documents for query: {q}")
        for doc, score in results:
            print(f"Document: {doc.page_content}, Score: {score}")

    # Save state
    retrieval_system.save_state("retrieval_state.npy")

    # Load state
    new_retrieval_system = AdvancedVectorRetrieval()
    new_retrieval_system.load_state("retrieval_state.npy")

    # Verify loaded state
    top_docs_after_loading = new_retrieval_system.retrieve(query, top_k=3)
    print("\nTop documents after loading state:")
    for doc, score in top_docs_after_loading:
        print(f"Document: {doc.page_content}, Score: {score}")

    # Update a document
    retrieval_system.update_document(0, "Paris is the capital of France.", {"updated": True})
    updated_doc = retrieval_system.get_document(0)
    
    print("\nUpdated document: ")
    print(f"Document: {updated_doc.page_content}, Metadata: {updated_doc.metadata}")

    # Remove a document
    retrieval_system.remove_document(1)
    
    try:
        removed_doc = retrieval_system.get_document(1)
    except IndexError as e:
        print("\nDocument at index 1 has been removed.")

example_usage()