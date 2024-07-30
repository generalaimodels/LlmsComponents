import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
import joblib
import logging
from typing import List, Dict, Union, Optional, Any, Tuple
from enum import Enum
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/agent_content_retrieve.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

class DistanceStrategy(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class AgentContentRetrieval:
    def __init__(
        self,
        embedding_model,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        batch_size: int = 32
    ) -> None:
        self.embeddings = embedding_model
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