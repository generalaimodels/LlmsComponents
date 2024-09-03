import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from rag_config import CustomFaissConfig

class CustomFaiss(FAISS):
    def __init__(
        self,
        embedding_function: Union[Callable[[str], List[float]], Embeddings],
        index: Any,
        docstore: InMemoryDocstore,
        index_to_docstore_id: Dict[int, str],
        *,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        **kwargs: Any
    ) -> None:
        super().__init__(
            embedding_function,
            index,
            docstore,
            index_to_docstore_id,
            relevance_score_fn=relevance_score_fn,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        return super().add_texts(texts, metadatas, ids, **kwargs)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        return super().add_embeddings(text_embeddings, metadatas, ids, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return super().similarity_search_with_score_by_vector(
            embedding, k, filter, fetch_k, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return super().similarity_search_with_score(query, k, filter, fetch_k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any
    ) -> List[Document]:
        return super().similarity_search_by_vector(embedding, k, filter, fetch_k, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any
    ) -> List[Document]:
        return super().similarity_search(query, k, filter, fetch_k, **kwargs)

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return super().max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> List[Document]:
        return super().max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> List[Document]:
        return super().max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, filter, **kwargs
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return super().delete(ids, **kwargs)

    def merge_from(self, target: FAISS) -> None:
        super().merge_from(target)

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        faiss.write_index(self.index, str(path / f"{index_name}.faiss"))

        with open(path / f"{index_name}.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings,
        index_name: str = "index",
        *,
        allow_dangerous_deserialization: bool = False,
        **kwargs: Any
    ) -> "CustomFaiss":
        path = Path(folder_path)

        with open(path / f"{index_name}.pkl", "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)

        index = faiss.read_index(str(path / f"{index_name}.faiss"))

        return cls(
            embeddings.embed_query,
            index,
            docstore,
            index_to_docstore_id,
            **kwargs
        )
        
