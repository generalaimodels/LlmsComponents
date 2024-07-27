from typing import List, Dict, Union, Optional, Callable,Tuple
import numpy as np
import torch
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class AdvancedHistoryManager:
    def __init__(
        self,
        embedding_model: Union[str, Callable],
        search_depth: int = 5,
        distance_strategy: str = "cosine",
    ):
        self.embedding_model = (
            HuggingFaceEmbeddings(model_name=embedding_model)
            if isinstance(embedding_model, str)
            else embedding_model
        )
        self.search_depth = search_depth
        self.distance_strategy = distance_strategy
        self.document_embeddings = None

    def process_documents(
        self,
        documents: Union[List[str], Dict[str, Union[List[str], str]]],
        metadata: Optional[Union[List[str], Dict[str, Union[List[str], str]]]] = None,
    ) -> List[Document]:
        processed_docs = []
        
        if isinstance(documents, dict):
            for key, value in documents.items():
                if isinstance(value, list):
                    for item in value:
                        doc = Document(page_content=item, metadata={"source": key})
                        if metadata and isinstance(metadata, dict) and key in metadata:
                            doc.metadata.update(metadata[key])
                        processed_docs.append(doc)
                else:
                    doc = Document(page_content=value, metadata={"source": key})
                    if metadata and isinstance(metadata, dict) and key in metadata:
                        doc.metadata.update(metadata[key])
                    processed_docs.append(doc)
        elif isinstance(documents, list):
            for idx, doc in enumerate(documents):
                metadata_item = metadata[idx] if metadata and isinstance(metadata, list) else None
                processed_docs.append(Document(page_content=doc, metadata=metadata_item))
        
        return processed_docs

    def compute_embeddings(self, documents: List[Document]) -> np.ndarray:
        embeddings = []
        for doc in tqdm(documents, desc="Computing embeddings"):
            embedding = self.embedding_model.embed_query(doc.page_content)
            embeddings.append(embedding)
        return np.array(embeddings)

    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        if self.distance_strategy == "cosine":
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(doc_embeddings),
                dim=1
            )
        elif self.distance_strategy == "euclidean":
            similarity = -torch.cdist(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(doc_embeddings)
            ).squeeze()
        else:
            raise ValueError(f"Unsupported distance strategy: {self.distance_strategy}")
        
        return similarity.numpy()

    def retrieve_relevant_content(
        self,
        query: str,
        documents: Union[List[str], Dict[str, Union[List[str], str]]],
        metadata: Optional[Union[List[str], Dict[str, Union[List[str], str]]]] = None,
    ) -> List[Tuple[Document, float]]:
        processed_docs = self.process_documents(documents, metadata)
        
        if self.document_embeddings is None:
            self.document_embeddings = self.compute_embeddings(processed_docs)
        
        query_embedding = self.embedding_model.embed_query(query)
        similarities = self.compute_similarity(query_embedding, self.document_embeddings)
        
        top_indices = np.argsort(similarities)[-self.search_depth:][::-1]
        
        return [(processed_docs[i], similarities[i]) for i in top_indices]

    def query(
        self,
        query: str,
        documents: Union[List[str], Dict[str, Union[List[str], str]]],
        metadata: Optional[Union[List[str], Dict[str, Union[List[str], str]]]] = None,
    ) -> List[Tuple[Document, float]]:
        return self.retrieve_relevant_content(query, documents, metadata)

from typing import List, Dict, Union, Optional, Callable, Tuple, Any
import numpy as np
import torch
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class AdvancedHistoryManager_upate:
    def __init__(
        self,
        embedding_model: Union[str, Callable],
        search_depth: int = 5,
        distance_strategy: str = "cosine",
        chunk_size: int = 1000,
        use_gpu: bool = False,
    ):
        self.embedding_model = (
            HuggingFaceEmbeddings(model_name=embedding_model)
            if isinstance(embedding_model, str)
            else embedding_model
        )
        self.search_depth = search_depth
        self.distance_strategy = distance_strategy
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.document_embeddings: Optional[torch.Tensor] = None

    def process_documents(
        self,
        documents: Union[List[str], Dict[str, Union[List[str], str]]],
        metadata: Optional[Union[List[str], Dict[str, Union[List[str], str]]]] = None,
    ) -> List[Document]:
        processed_docs = []
        
        if isinstance(documents, dict):
            for key, value in documents.items():
                if isinstance(value, list):
                    for item in value:
                        doc = Document(page_content=item, metadata={"source": key})
                        if metadata and isinstance(metadata, dict) and key in metadata:
                            doc.metadata.update(metadata[key])
                        processed_docs.append(doc)
                else:
                    doc = Document(page_content=value, metadata={"source": key})
                    if metadata and isinstance(metadata, dict) and key in metadata:
                        doc.metadata.update(metadata[key])
                    processed_docs.append(doc)
        elif isinstance(documents, list):
            for idx, doc in enumerate(documents):
                metadata_item = metadata[idx] if metadata and isinstance(metadata, list) else None
                processed_docs.append(Document(page_content=doc, metadata=metadata_item))
        
        return processed_docs

    def compute_embeddings(self, documents: List[Document]) -> torch.Tensor:
        embeddings = []
        for i in tqdm(range(0, len(documents), self.chunk_size), desc="Computing embeddings"):
            chunk = documents[i:i+self.chunk_size]
            chunk_embeddings = self.embedding_model.embed_documents([doc.page_content for doc in chunk])
            embeddings.extend(chunk_embeddings)
        return torch.tensor(embeddings, device=self.device)

    def compute_similarity(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        if self.distance_strategy == "cosine":
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                doc_embeddings,
                dim=1
            )
        elif self.distance_strategy == "euclidean":
            similarity = -torch.cdist(
                query_embedding.unsqueeze(0),
                doc_embeddings
            ).squeeze()
        else:
            raise ValueError(f"Unsupported distance strategy: {self.distance_strategy}")
        
        return similarity

    def retrieve_relevant_content(
        self,
        query: str,
        documents: Union[List[str], Dict[str, Union[List[str], str]]],
        metadata: Optional[Union[List[str], Dict[str, Union[List[str], str]]]] = None,
    ) -> List[Tuple[Document, float]]:
        processed_docs = self.process_documents(documents, metadata)
        
        if self.document_embeddings is None:
            self.document_embeddings = self.compute_embeddings(processed_docs)
        
        query_embedding = torch.tensor(self.embedding_model.embed_query(query), device=self.device)
        similarities = self.compute_similarity(query_embedding, self.document_embeddings)
        
        top_indices = torch.argsort(similarities, descending=True)[:self.search_depth]
        
        return [(processed_docs[i], similarities[i].item()) for i in top_indices]

    def query(
        self,
        query: str,
        documents: Union[List[str], Dict[str, Union[List[str], str]]],
        metadata: Optional[Union[List[str], Dict[str, Union[List[str], str]]]] = None,
    ) -> List[Tuple[Document, float]]:
        return self.retrieve_relevant_content(query, documents, metadata)

    def update_documents(
        self,
        new_documents: Union[List[str], Dict[str, Union[List[str], str]]],
        new_metadata: Optional[Union[List[str], Dict[str, Union[List[str], str]]]] = None,
    ) -> None:
        processed_new_docs = self.process_documents(new_documents, new_metadata)
        new_embeddings = self.compute_embeddings(processed_new_docs)
        
        if self.document_embeddings is None:
            self.document_embeddings = new_embeddings
        else:
            self.document_embeddings = torch.cat([self.document_embeddings, new_embeddings], dim=0)

    def clear_cache(self) -> None:
        self.document_embeddings = None
        torch.cuda.empty_cache()
# Example usage
if __name__ == "__main__":
    history_manager = AdvancedHistoryManager_upate(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    
    documents = {
        "doc1": ["This is the first document.", "It contains two sentences."],
        "doc2": "This is the second document.",
        "doc3": ["This is the third document.", "It also has multiple sentences.", "Three in total."],
    }
    
    metadata = {
        "doc1": {"author": "Alice", "date": "2023-01-01"},
        "doc2": {"author": "Bob", "date": "2023-01-02"},
        "doc3": {"author": "Charlie", "date": "2023-01-03"},
    }
    
    query = "Which document has multiple sentences?"
    
    results = history_manager.query(query, documents, metadata)
    
    for doc, score in results:
        print(f"Document: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Similarity Score: {score}")
        print("---")