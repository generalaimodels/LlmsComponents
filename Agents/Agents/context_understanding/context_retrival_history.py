from typing import List, Dict, Union, Optional, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdvancedVectorRetrieval:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        max_workers: int = 4
    ) -> None:
        """
        Initialize the AdvancedVectorRetrieval system.

        Args:
            embedding_model (str): The name of the Hugging Face model to use for embeddings.
            distance_strategy (DistanceStrategy): The strategy to use for calculating distances.
            max_workers (int): Maximum number of threads for parallel processing.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.distance_strategy = distance_strategy
        self.documents: List[Document] = []
        self.document_embeddings: List[np.ndarray] = []
        self.max_workers = max_workers

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

        new_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadata or [{}] * len(documents))
        ]
        
        self.documents.extend(new_docs)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_embeddings = list(executor.map(self.embeddings.embed_query, [doc.page_content for doc in new_docs]))
        
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
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.retrieve, query, top_k, depth_search) for query in queries]
            return [future.result() for future in tqdm(as_completed(futures), total=len(queries), desc="Processing queries")]

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
        joblib.dump(state, file_path)

    def load_state(self, file_path: str) -> None:
        """
        Load the state of the retrieval system from a file.

        Args:
            file_path (str): The path to the file from which the state will be loaded.
        """
        state = joblib.load(file_path)
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

    def search_by_metadata(self, metadata_query: Dict[str, Any]) -> List[Document]:
        """
        Search for documents based on metadata.

        Args:
            metadata_query (Dict[str, Any]): A dictionary of metadata key-value pairs to search for.

        Returns:
            List[Document]: A list of documents that match the metadata query.
        """
        return [doc for doc in self.documents if all(doc.metadata.get(k) == v for k, v in metadata_query.items())]

    def get_similar_documents(self, document_index: int, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Find documents similar to a given document in the retrieval system.

        Args:
            document_index (int): The index of the document to find similarities for.
            top_k (int): The number of top similar documents to return.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing similar documents and their similarity scores.
        """
        if 0 <= document_index < len(self.documents):
            query_embedding = self.document_embeddings[document_index]
            similarities = self._calculate_similarity(query_embedding)
            
            # Exclude the query document itself
            similarities[document_index] = -np.inf
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(self.documents[idx], float(similarities[idx])) for idx in top_indices]
        else:
            raise IndexError("Document index out of range")

    def update_embeddings_model(self, new_model_name: str) -> None:
        """
        Update the embeddings model used by the retrieval system.

        Args:
            new_model_name (str): The name of the new Hugging Face model to use for embeddings.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=new_model_name)
        
        # Re-embed all documents with the new model
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.document_embeddings = list(executor.map(self.embeddings.embed_query, [doc.page_content for doc in self.documents]))

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the retrieval system.

        Returns:
            Dict[str, Any]: A dictionary containing system statistics.
        """
        return {
            "total_documents": len(self.documents),
            "embedding_model": self.embeddings.model_name,
            "distance_strategy": self.distance_strategy,
            "unique_metadata_keys": set().union(*(doc.metadata.keys() for doc in self.documents))
        }

    def clear_all_documents(self) -> None:
        """
        Remove all documents from the retrieval system.
        """
        self.documents.clear()
        self.document_embeddings.clear()

    def bulk_update_metadata(self, updates: List[Tuple[int, Dict[str, Any]]]) -> None:
        """
        Update metadata for multiple documents in bulk.

        Args:
            updates (List[Tuple[int, Dict[str, Any]]]): List of tuples containing document index and new metadata.
        """
        for index, new_metadata in updates:
            if 0 <= index < len(self.documents):
                self.documents[index].metadata.update(new_metadata)
            else:
                raise IndexError(f"Document index {index} out of range")

    def retrieve_with_history(
        self,
        query: str,
        history: List[str],
        top_k: int = 5,
        depth_search: int = 1,
        history_weight: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents considering both the current query and query history.

        Args:
            query (str): The current query string.
            history (List[str]): List of previous queries.
            top_k (int): The number of top results to return.
            depth_search (int): The depth of the search for re-ranking.
            history_weight (float): Weight given to historical queries (0 to 1).

        Returns:
            List[Tuple[Document, float]]: List of tuples containing retrieved documents and their relevance scores.
        """
        current_query_embedding = self.embeddings.embed_query(query)
        
        # Embed historical queries
        history_embeddings = [self.embeddings.embed_query(h) for h in history]
        
        # Combine current query with historical queries
        combined_query = np.average([current_query_embedding] + history_embeddings, 
                                    axis=0, 
                                    weights=[1 - history_weight] + [history_weight / len(history)] * len(history))
        
        similarities = self._calculate_similarity(combined_query)
        
        top_indices = np.argsort(similarities)[-top_k * depth_search:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc_query_similarity = similarities[idx]
            
            doc_embedding = self.document_embeddings[idx]
            relevance_score = (doc_query_similarity + cosine_similarity([combined_query], [doc_embedding])[0][0]) / 2
            
            results.append((doc, float(relevance_score)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

from typing import List, Dict, Union, Optional, Any, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class AdvancedVectorRetrieval_gpt4:
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
            documents: A list of strings, a dictionary of strings to list of strings, or a single string.
            metadata: Optional metadata for each document.
        """
        if isinstance(documents, str):
            documents = [documents]
        elif isinstance(documents, dict):
            documents = [f"{k}: {' '.join(v)}" for k, v in documents.items()]

        new_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadata or [{}] * len(documents))
        ]

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
        for idx in tqdm(top_indices, desc="Re-ranking results"):
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
    
    def add_metadata(self, index: int, metadata: Dict[str, Any]) -> None:
        """
        Add or update metadata for a specific document in the retrieval system.

        Args:
            index (int): The index of the document for which metadata is to be added or updated.
            metadata (Dict[str, Any]): The metadata to add or update for the document.
        """
        if 0 <= index < len(self.documents):
            self.documents[index].metadata.update(metadata)
        else:
            raise IndexError("Document index out of range")

    def clear_documents(self) -> None:
        """
        Clear all documents and their embeddings from the retrieval system.
        """
        self.documents.clear()
        self.document_embeddings.clear()

    def search_by_metadata(self, metadata_query: Dict[str, Any]) -> List[Document]:
        """
        Search for documents that match the given metadata query.

        Args:
            metadata_query (Dict[str, Any]): Metadata query to search for matching documents.

        Returns:
            List[Document]: List of documents that match the metadata query.
        """
        matching_documents = [
            doc for doc in self.documents
            if all(item in doc.metadata.items() for item in metadata_query.items())
        ]
        return matching_documents

    def document_count(self) -> int:
        """
        Get the number of documents in the retrieval system.

        Returns:
            int: The number of documents.
        """
        return len(self.documents)

    def retrieve_similar_documents(
        self, document_content: str, top_k: int = 5, depth_search: int = 1
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents similar to the provided document content.

        Args:
            document_content (str): The content of the document to find similar documents to.
            top_k (int): The number of top results to return.
            depth_search (int): The depth of the search for re-ranking.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing the similar documents and their relevance scores.
        """
        content_embedding = self.embeddings.embed_query(document_content)
        similarities = self._calculate_similarity(content_embedding)

        top_indices = np.argsort(similarities)[-top_k * depth_search:][::-1]

        results = []
        for idx in tqdm(top_indices, desc="Re-ranking results"):
            doc = self.documents[idx]
            doc_query_similarity = similarities[idx]

            # Re-rank based on document-query similarity
            doc_embedding = self.document_embeddings[idx]
            relevance_score = (doc_query_similarity + cosine_similarity([content_embedding], [doc_embedding])[0][0]) / 2

            results.append((doc, float(relevance_score)))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def __str__(self) -> str:
        """
        Return a string representation of the retrieval system, summarizing the number of documents.

        Returns:
            str: String representation of the retrieval system.
        """
        return f"AdvancedVectorRetrieval with {len(self.documents)} documents."

# Example usage:
retrieval_system = AdvancedVectorRetrieval_gpt4()
retrieval_system.add_documents(["This is a sample document.", "Another document."])
results = retrieval_system.retrieve("sample query")
for doc, score in results:
    print(f"Content: {doc.page_content}, Score: {score}")