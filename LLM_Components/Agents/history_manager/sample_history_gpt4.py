from typing import List, Dict, Union, Optional,Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
import numpy as np
import torch  # For similarity distance function
from tqdm import tqdm


class HistoryManager:
    def __init__(
        self,
        documents: Union[List[str], Dict[str, Union[str, List[str]]]],
        metadata: Union[List[str], Dict[str, Union[str, List[str]]]],
        embedding_model: Union[str, callable],
        search_depth: int = 5
    ) -> None:
        self.documents = documents
        self.metadata = metadata
        self.search_depth = search_depth
        self.embedding_model = self.load_embedding_model(embedding_model)
        self.document_embeddings = self.compute_embeddings(documents)

    def load_embedding_model(self, model: Union[str, callable]) -> callable:
        if isinstance(model, str):
            return HuggingFaceEmbeddings(model)
        elif callable(model):
            return model
        else:
            raise ValueError("Invalid embedding model type.")

    def compute_embeddings(self, data: Union[List[str], Dict[str, Union[str, List[str]]]]) -> np.ndarray:
        if isinstance(data, dict):
            # Flatten the dictionary to extract strings
            data = [item for sublist in data.values() for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        embeddings = []
        for text in tqdm(data, desc="Computing Embeddings"):
            embedding = self.embedding_model.encode(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)

    def query(self, user_query: str) -> List[Tuple[str, float]]:
        query_embedding = self.embedding_model.encode(user_query)
        similarities = self.compute_similarities(query_embedding)
        ranked_documents = self.rank_documents(similarities)
        return ranked_documents

    def compute_similarities(self, query_embedding: np.ndarray) -> List[float]:
        if torch.cuda.is_available():
            query_embedding = torch.tensor(query_embedding).cuda()
            document_embeddings = torch.tensor(self.document_embeddings).cuda()
        else:
            query_embedding = torch.tensor(query_embedding)
            document_embeddings = torch.tensor(self.document_embeddings)

        similarities = torch.nn.functional.cosine_similarity(document_embeddings, query_embedding.unsqueeze(0))
        return similarities.cpu().numpy()

    def rank_documents(self, similarities: List[float]) -> List[Tuple[str, float]]:
        ranked_indices = np.argsort(similarities)[::-1][:self.search_depth]
        ranked_documents = [(self.documents[idx], similarities[idx]) for idx in ranked_indices]
        return ranked_documents


# Example usage
if __name__ == "__main__":
    documents = ["Doc1 content", "Doc2 content", "Doc3 content"]
    metadata = {"Doc1": "Meta1", "Doc2": "Meta2", "Doc3": "Meta3"}
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    history_manager = HistoryManager(
        documents=documents,
        metadata=metadata,
        embedding_model=embedding_model_name,
        search_depth=3
    )
    
    query = "Relevant content query"
    results = history_manager.query(query)
    for doc, score in results:
        print(f"Document: {doc}, Score: {score}")