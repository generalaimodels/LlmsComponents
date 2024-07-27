from typing import List, Dict, Union, Optional, Any, Tuple
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document

class AdvancedVectorRetrieval:
    """Advanced Vector Retrieval Class using FAISS and HuggingFace embeddings."""

    def __init__(
        self,
        embedding_model: HuggingFaceEmbeddings,
        distance_type: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        dipth_search: int = 4,
    ) -> None:
        """
        Initialize the AdvancedVectorRetrieval instance.

        Args:
            embedding_model (HuggingFaceEmbeddings): The embedding model to use.
            distance_type (DistanceStrategy, optional): The distance strategy to use for retrieval.
            dipth_search (int, optional): Number of documents to retrieve.
        """
        self.embedding_model = embedding_model
        self.distance_type = distance_type
        self.dipth_search = dipth_search
        self.faiss_store = None

    def _prepare_documents(self, documents: Union[List[str], Dict[str, List[str]], str]) -> List[Document]:
        """
        Prepare documents for storage.

        Args:
            documents (Union[List[str], Dict[str, List[str]], str]): Documents in different formats.

        Returns:
            List[Document]: List of Document objects.
        """
        if isinstance(documents, str):
            return [Document(page_content=documents, metadata={})]
        elif isinstance(documents, list):
            return [Document(page_content=doc, metadata={}) for doc in documents]
        elif isinstance(documents, dict):
            return [Document(page_content=key, metadata={'tags': value}) for key, value in documents.items()]
        else:
            raise TypeError("documents must be str, list of str, or dict of str to list of str.")

    def add_documents(self, documents: Union[List[str], Dict[str, List[str]], str]) -> List[str]:
        """
        Add documents to the FAISS vector store.

        Args:
            documents (Union[List[str], Dict[str, List[str]], str]): Documents to add.

        Returns:
            List[str]: List of IDs assigned to the added documents.
        """
        prepared_docs = self._prepare_documents(documents)
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in prepared_docs])
        
        if self.faiss_store is None:
            self.faiss_store = FAISS(self.embedding_model, index=None, docstore=None, index_to_docstore_id={})

        ids = self.faiss_store.add_texts(
            texts=[doc.page_content for doc in prepared_docs],
            metadatas=[doc.metadata for doc in prepared_docs]
        )
        return ids

    def retrieve_similar_documents(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve similar documents based on the query.

        Args:
            query (str): The query string for similarity search.
            k (Optional[int]): Number of documents to retrieve. Defaults to dipth_search.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing Document and similarity score.
        """
        if self.faiss_store is None:
            raise RuntimeError("FAISS store is not initialized. Please add documents first.")

        query_embedding = self.embedding_model.embed_query(query)
        k = k or self.dipth_search

        results = self.faiss_store.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            k=k,
            filter=None,  # You can add filtering functionality here
            fetch_k=k * 2  # Fetch more to ensure we meet k after filtering if needed
        )
        return results


from typing import List, Dict, Union, Optional, Tuple
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

# Example usage:
if __name__ == "__main__":
    retriever = AdvancedVectorRetrieval()
    
    # Add documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
    ]
    metadata = [
        {"source": "proverb", "language": "english"},
        {"source": "quote", "author": "Lao Tzu"},
        {"source": "literature", "author": "William Shakespeare"},
    ]
    retriever.add_documents(documents, metadata)
    
    # Perform retrieval
    query = "What is the meaning of life?"
    results = retriever.retrieve(query, top_k=2, depth_search=2)
    
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()