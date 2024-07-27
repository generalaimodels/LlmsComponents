from typing import List, Dict, Union, Optional, Tuple, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class AdvancedVectorRetrieval:
    def __init__(
        self,
        embedding_model: str = "distilbert-base-uncased",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the AdvancedVectorRetrieval system.

        Args:
            embedding_model (str): The name of the Hugging Face model to use for embeddings.
            distance_strategy (DistanceStrategy): The strategy to use for calculating distances.
            batch_size (int): The batch size for processing embeddings.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.distance_strategy = distance_strategy
        self.batch_size = batch_size
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

        new_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadata or [{}] * len(documents))
        ]

        self.documents.extend(new_docs)

        new_embeddings = self._batch_embed_documents([doc.page_content for doc in new_docs])
        self.document_embeddings.extend(new_embeddings)

    def _batch_embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed documents in batches for improved performance.

        Args:
            texts (List[str]): List of document texts to embed.

        Returns:
            List[np.ndarray]: List of document embeddings.
        """
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings.extend(self.embeddings.embed_documents(batch))
        return embeddings

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
            return -euclidean_distances([query_embedding], self.document_embeddings)[0]
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

        top_indices = np.argsort(similarities)[-top_k * depth_search :][::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc_query_similarity = similarities[idx]

            # Re-rank based on document-query similarity
            doc_embedding = self.document_embeddings[idx]
            relevance_score = (
                doc_query_similarity + cosine_similarity([query_embedding], [doc_embedding])[0][0]
            ) / 2

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
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.retrieve, query, top_k, depth_search) for query in queries
            ]
            results = [future.result() for future in tqdm(as_completed(futures), total=len(queries), desc="Processing queries")]
        return results

    def update_document(
        self, index: int, new_content: str, new_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
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

    def clear_documents(self) -> None:
        """Clear all documents and embeddings from the retrieval system."""
        self.documents.clear()
        self.document_embeddings.clear()

    def get_document_count(self) -> int:
        """
        Get the number of documents in the retrieval system.

        Returns:
            int: The number of documents.
        """
        return len(self.documents)

    def get_document(self, index: int) -> Optional[Document]:
        """
        Retrieve a specific document by index.

        Args:
            index (int): The index of the document to retrieve.

        Returns:
            Optional[Document]: The requested document, or None if the index is out of range.
        """
        if 0 <= index < len(self.documents):
            return self.documents[index]
        return None

    def search_by_metadata(
        self, metadata_filter: Dict[str, Any], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents based on metadata filters.

        Args:
            metadata_filter (Dict[str, Any]): A dictionary of metadata key-value pairs to filter by.
            top_k (int): The maximum number of results to return.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing the matching documents and their relevance scores.
        """
        matching_docs = []
        for idx, doc in enumerate(self.documents):
            if all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                matching_docs.append((doc, 1.0))  # Use 1.0 as a placeholder score

        return sorted(matching_docs, key=lambda x: x[1], reverse=True)[:top_k]

    def hybrid_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        depth_search: int = 1,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a hybrid search combining semantic similarity and metadata filtering.

        Args:
            query (str): The query string for semantic search.
            metadata_filter (Optional[Dict[str, Any]]): Optional metadata filter.
            top_k (int): The number of top results to return.
            depth_search (int): The depth of the search for re-ranking.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing the retrieved documents and their relevance scores.
        """
        semantic_results = self.retrieve(query, top_k=top_k * 2, depth_search=depth_search)
        
        if metadata_filter:
            filtered_results = [
                (doc, score) for doc, score in semantic_results
                if all(doc.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            filtered_results = semantic_results

        return filtered_results[:top_k]

    def save_index(self, file_path: str) -> None:
        """
        Save the current index (documents and embeddings) to a file.

        Args:
            file_path (str): The path to save the index file.
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump((self.documents, self.document_embeddings), f)

    @classmethod
    def load_index(cls, file_path: str, embedding_model: str, distance_strategy: DistanceStrategy) -> 'AdvancedVectorRetrieval':
        """
        Load an index from a file and create a new AdvancedVectorRetrieval instance.

        Args:
            file_path (str): The path to the saved index file.
            embedding_model (str): The name of the Hugging Face model to use for embeddings.
            distance_strategy (DistanceStrategy): The strategy to use for calculating distances.

        Returns:
            AdvancedVectorRetrieval: A new instance with the loaded index.
        """
        import pickle
        with open(file_path, 'rb') as f:
            documents, document_embeddings = pickle.load(f)

        instance = cls(embedding_model, distance_strategy)
        instance.documents = documents
        instance.document_embeddings = document_embeddings
        return instance

    def update_embeddings_model(self, new_model: str) -> None:
        """
        Update the embeddings model and recalculate all document embeddings.

        Args:
            new_model (str): The name of the new Hugging Face model to use for embeddings.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=new_model)
        self.document_embeddings = self._batch_embed_documents([doc.page_content for doc in self.documents])

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dict[str, Any]: A dictionary containing index statistics.
        """
        return {
            "total_documents": len(self.documents),
            "embedding_dimensions": len(self.document_embeddings[0]) if self.document_embeddings else 0,
            "distance_strategy": self.distance_strategy.value,
            "embedding_model": self.embeddings.model_name,
        }


from typing import List, Dict, Union, Optional, Any, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class AdvancedVectorRetrieval_GPT4:
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

if __name__ == "__main__":
    # Example usage
    retrieval_system = AdvancedVectorRetrieval_GPT4()
    
    docs = ["This is a sample document.", "Another document for testing.", "More documents to test the system."]
    retrieval_system.add_documents(docs)
    
    query = "sample"
    results = retrieval_system.retrieve(query)
    
    for doc, score in results:
        print(f"Document: {doc.page_content}, Score: {score}")
    
    # Save state to a file
    retrieval_system.save_state("retrieval_state.npy")
    
    # Load state from a file
    loaded_retrieval_system = AdvancedVectorRetrieval_GPT4()
    loaded_retrieval_system.load_state("retrieval_state.npy")
    
    # Check if loaded state is the same
    loaded_results = loaded_retrieval_system.retrieve(query)
    for doc, score in loaded_results:
        print(f"Loaded Document: {doc.page_content}, Score: {score}")