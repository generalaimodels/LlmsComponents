import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
import joblib
from typing import List, Dict, Union, Optional, Any, Tuple
from enum import Enum
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
class DistanceStrategy(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class AdvancedVectorRetrieval:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        batch_size: int = 32
    ) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.distance_strategy = distance_strategy
        self.batch_size = batch_size
        self.documents: List[Document] = []
        self.document_embeddings: np.ndarray = np.array([])

    def add_documents(
        self,
        documents: Union[List[str], Dict[str, List[str]], str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
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
        self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings]) if self.document_embeddings.size else new_embeddings

    def _batch_embed_documents(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding documents"):
            batch = texts[i:i + self.batch_size]
            embeddings.extend(self.embeddings.embed_documents(batch))
        return np.array(embeddings)

    def _calculate_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
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
        query_embeddings = self._batch_embed_documents(queries)
        
        all_results = []
        for query_embedding in tqdm(query_embeddings, desc="Processing queries"):
            similarities = self._calculate_similarity(query_embedding)
            top_indices = np.argsort(similarities)[-top_k * depth_search:][::-1]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                doc_query_similarity = similarities[idx]
                
                doc_embedding = self.document_embeddings[idx]
                relevance_score = (doc_query_similarity + cosine_similarity([query_embedding], [doc_embedding])[0][0]) / 2
                
                results.append((doc, float(relevance_score)))
            
            all_results.append(sorted(results, key=lambda x: x[1], reverse=True)[:top_k])
        
        return all_results

    def update_document(self, index: int, new_content: str, new_metadata: Optional[Dict[str, Any]] = None) -> None:
        if 0 <= index < len(self.documents):
            self.documents[index] = Document(page_content=new_content, metadata=new_metadata or {})
            self.document_embeddings[index] = self.embeddings.embed_query(new_content)
        else:
            raise IndexError("Document index out of range")

    def remove_document(self, index: int) -> None:
        if 0 <= index < len(self.documents):
            del self.documents[index]
            self.document_embeddings = np.delete(self.document_embeddings, index, axis=0)
        else:
            raise IndexError("Document index out of range")

    def save_state(self, file_path: str) -> None:
        state = {
            "documents": [(doc.page_content, doc.metadata) for doc in self.documents],
            "document_embeddings": self.document_embeddings,
            "embedding_model": self.embeddings.model_name,
            "distance_strategy": self.distance_strategy,
            "batch_size": self.batch_size,
        }
        joblib.dump(state, file_path)

    def load_state(self, file_path: str) -> None:
        state = joblib.load(file_path)
        self.documents = [Document(page_content=content, metadata=meta) for content, meta in state["documents"]]
        self.document_embeddings = state["document_embeddings"]
        self.embeddings = HuggingFaceEmbeddings(model_name=state["embedding_model"])
        self.distance_strategy = state["distance_strategy"]
        self.batch_size = state["batch_size"]

    def get_document(self, index: int) -> Document:
        if 0 <= index < len(self.documents):
            return self.documents[index]
        else:
            raise IndexError("Document index out of range")
# Test the implementation with dummy data
if __name__ == "__main__":
    # Initialize the retrieval system
    retriever = AdvancedVectorRetrieval()

    # Add dummy documents
    dummy_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "To be or not to be, that is the question.",
        "In a galaxy far, far away...",
        "It was the best of times, it was the worst of times."
    ]
    dummy_metadata = [
        {"source": "example1.txt"},
        {"source": "example2.txt"},
        {"source": "example3.txt"},
        {"source": "example4.txt"},
        {"source": "example5.txt"}
    ]

    retriever.add_documents(dummy_docs, dummy_metadata)

    # Test single retrieval
    query = "What does the fox do?"
    results = retriever.retrieve(query, top_k=2)
    print(f"\nQuery: {query}")
    for doc, score in results:
        print(f"Document: {doc.page_content}")
        print(f"Score: {score}")
        print(f"Metadata: {doc.metadata}")
        print()

    # Test batch retrieval
    queries = [
        "What is the meaning of life?",
        "Tell me about space",
        "Describe the best and worst times"
    ]
    batch_results = retriever.batch_retrieve(queries, top_k=2)
    for i, query_results in enumerate(batch_results):
        print(f"\nQuery: {queries[i]}")
        for doc, score in query_results:
            print(f"Document: {doc.page_content}")
            print(f"Score: {score}")
            print(f"Metadata: {doc.metadata}")
            print()

    # Test document update
    new_content = "The lazy dog jumps over the quick brown fox."
    retriever.update_document(0, new_content, {"source": "updated_example1.txt"})
    print("\nUpdated document:")
    print(retriever.get_document(0).page_content)
    print(retriever.get_document(0).metadata)

    # Test document removal
    retriever.remove_document(1)
    print(f"\nNumber of documents after removal: {len(retriever.documents)}")

    # Test save and load state
    retriever.save_state("retriever_state.joblib")
    new_retriever = AdvancedVectorRetrieval()
    new_retriever.load_state("retriever_state.joblib")
    
    print("\nState loaded. Performing a query on the loaded retriever:")
    results = new_retriever.retrieve("What does the fox do?", top_k=1)
    for doc, score in results:
        print(f"Document: {doc.page_content}")
        print(f"Score: {score}")
        print(f"Metadata: {doc.metadata}")

    # Test error handling
    try:
        retriever.get_document(10)
    except IndexError as e:
        print(f"\nExpected error caught: {e}")

    try:
        retriever.update_document(10, "This should fail")
    except IndexError as e:
        print(f"Expected error caught: {e}")

    try:
        retriever.remove_document(10)
    except IndexError as e:
        print(f"Expected error caught: {e}")

print("\nAll tests completed.")