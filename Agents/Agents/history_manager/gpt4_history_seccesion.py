import logging
from typing import List, Dict, Union, Callable, Any
import numpy as np
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HistorySession:
    def __init__(self, embedding_model: Callable):
        self.history = []
        self.embedding_model = embedding_model
        self.index = None

    def add_history(self, content: Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]):
        self.history.append(content)
        logging.info(f'Added new content to history. Current history count: {self.get_history_count()}')

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def retrieve(self, query: str, top_k: int = 5) -> Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]:
        query_embedding = self.embedding_model.embed_query(query)
        similarities = []
        for content in self.history:
            content_embedding = self.embedding_model.embed_query(str(content))
            similarity = self.calculate_similarity(query_embedding, content_embedding)
            similarities.append((similarity, content))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in similarities[:top_k]]

    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]]:
        return [self.retrieve(query, top_k) for query in queries]

    def update_history(self, index: int, new_content: Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]):
        if 0 <= index < len(self.history):
            self.history[index] = new_content
            logging.info(f'Updated content at index {index}.')
        else:
            logging.error(f'Index {index} out of range. Cannot update history.')
            raise IndexError("Index out of range.")

    def remove_history(self, index: int):
        if 0 <= index < len(self.history):
            del self.history[index]
            logging.info(f'Removed content from index {index}. Current history count: {self.get_history_count()}')
        else:
            logging.error(f'Index {index} out of range. Cannot remove history.')
            raise IndexError("Index out of range.")

    def get_history_count(self) -> int:
        return len(self.history)

    def get_history_id(self, index: int) -> Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]:
        if 0 <= index < len(self.history):
            return self.history[index]
        else:
            logging.error(f'Index {index} out of range. Cannot retrieve history id.')
            raise IndexError("Index out of range.")

    def get_history_index(self, content: Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]) -> int:
        try:
            return self.history.index(content)
        except ValueError:
            logging.error('Content not found in history. Cannot retrieve history index.')
            raise ValueError("Content not found in history.")

    def hybrid_search(self, query: str, top_k: int = 5) -> Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]:
        # Implementing a hybrid search combining multiple criteria
        # Here, we might combine semantic similarity with keyword matching, for example
        query_embedding = self.embedding_model.embed_query(query)
        similarities = []
        for content in self.history:
            content_embedding = self.embedding_model.embed_query(str(content))
            similarity = self.calculate_similarity(query_embedding, content_embedding)
            keyword_match_score = self.keyword_match_score(query, str(content))
            hybrid_score = (similarity + keyword_match_score) / 2  # Simple average for demonstration
            similarities.append((hybrid_score, content))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in similarities[:top_k]]

    def keyword_match_score(self, query: str, content: str) -> float:
        query_keywords = set(query.lower().split())
        content_keywords = set(content.lower().split())
        common_keywords = query_keywords.intersection(content_keywords)
        return len(common_keywords) / len(query_keywords) if query_keywords else 0.0

    def save_index(self, file_path: str):
        torch.save(self.index, file_path)
        logging.info(f'Index saved to {file_path}.')

    def load_index(self, file_path: str):
        self.index = torch.load(file_path)
        logging.info(f'Index loaded from {file_path}.')

# Example usage:
if __name__ == "__main__":
    # Initialize the embedding model (for demonstration, using a dummy callable)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # force_download=False
    )

    session = HistorySession(embedding_model)

    # Adding sample history
    session.add_history("This is a test history.")
    session.add_history(["Another entry", "With multiple strings."])
    session.add_history({"key": ["value1", "value2"]})

    # Retrieve sample query
    print(session.retrieve("test query"))

    # Batch retrieve sample queries
    queries = ["test query 1", "test query 2"]
    print(session.batch_retrieve(queries))

    # Update a history entry
    session.update_history(1, "Updated entry")

    # Remove a history entry
    session.remove_history(0)

    # Get history count
    print(f'History count: {session.get_history_count()}')

    # Get history by index
    print(session.get_history_id(0))

    # Get index of specific content
    try:
        index = session.get_history_index("Updated entry")
        print(f'Content index: {index}')
    except ValueError as e:
        print(e)

    # Perform hybrid search
    print(session.hybrid_search("test query"))