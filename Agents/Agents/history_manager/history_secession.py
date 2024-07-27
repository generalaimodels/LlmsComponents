from typing import List, Dict, Union, Optional, Callable, Tuple
import numpy as np
import torch
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import faiss
import json
import os


class HistorySession:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.history: List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]] = []
        self.embeddings: List[np.ndarray] = []
        self.index = faiss.IndexFlatL2(384)  # Assuming 384-dimensional embeddings
        self.id_to_index: Dict[int, int] = {}

    def add_history(self, query: str, response: Union[str, List[str], Dict[str, List[str]]]) -> None:
        """Add a new entry to the history."""
        history_id = len(self.history)
        self.history.append({"query": query, "response": response})
        embedding = self.embedding_model.embed_query(query)
        self.embeddings.append(embedding)
        self.index.add(np.array([embedding]))
        self.id_to_index[history_id] = len(self.embeddings) - 1

    def calculate_similarity(self, query: str) -> np.ndarray:
        """Calculate similarity between the query and all history entries."""
        query_embedding = self.embedding_model.embed_query(query)
        return np.dot(self.embeddings, query_embedding)

    def retrieve(self, query: str, k: int = 1) -> List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Retrieve the top k most similar history entries."""
        query_embedding = self.embedding_model.embed_query(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.history[self.id_to_index[i]] for i in indices[0]]

    def batch_retrieve(self, queries: List[str], k: int = 1) -> List[List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]]:
        """Batch retrieve the top k most similar history entries for multiple queries."""
        query_embeddings = [self.embedding_model.embed_query(query) for query in queries]
        distances, indices = self.index.search(np.array(query_embeddings), k)
        return [[self.history[self.id_to_index[i]] for i in query_indices] for query_indices in indices]

    def update_history(self, history_id: int, query: str, response: Union[str, List[str], Dict[str, List[str]]]) -> None:
        """Update an existing history entry."""
        if history_id < len(self.history):
            self.history[history_id] = {"query": query, "response": response}
            new_embedding = self.embedding_model.embed_query(query)
            self.embeddings[self.id_to_index[history_id]] = new_embedding
            self.index.remove_ids(np.array([history_id]))
            self.index.add(np.array([new_embedding]))

    def remove_history(self, history_id: int) -> None:
        """Remove a history entry."""
        if history_id < len(self.history):
            del self.history[history_id]
            index_to_remove = self.id_to_index[history_id]
            del self.embeddings[index_to_remove]
            self.index.remove_ids(np.array([history_id]))
            del self.id_to_index[history_id]
            # Update id_to_index mapping
            for i in range(history_id + 1, len(self.history) + 1):
                if i in self.id_to_index:
                    self.id_to_index[i - 1] = self.id_to_index[i] - 1
                    del self.id_to_index[i]

    def get_history_count(self) -> int:
        """Get the total number of history entries."""
        return len(self.history)

    def get_history_by_id(self, history_id: int) -> Optional[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Get a history entry by its ID."""
        return self.history[history_id] if history_id < len(self.history) else None

    def get_history_index(self, history_id: int) -> Optional[int]:
        """Get the index of a history entry by its ID."""
        return self.id_to_index.get(history_id)

    def hybrid_search(self, query: str, k: int = 1, alpha: float = 0.5) -> List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Perform a hybrid search combining embedding similarity and keyword matching."""
        embedding_results = self.retrieve(query, k)
        keyword_results = self._keyword_search(query, k)
        
        combined_results = {}
        for i, result in enumerate(embedding_results):
            combined_results[i] = (1 - alpha) * (k - i) / k

        for i, result in enumerate(keyword_results):
            if i in combined_results:
                combined_results[i] += alpha * (k - i) / k
            else:
                combined_results[i] = alpha * (k - i) / k

        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        return [self.history[i] for i, _ in sorted_results[:k]]

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Perform a simple keyword-based search."""
        query_words = set(query.lower().split())
        scores = []
        for entry in self.history:
            entry_words = set(entry["query"].lower().split())
            score = len(query_words.intersection(entry_words)) / len(query_words)
            scores.append(score)
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.history[i] for i in top_k_indices]

    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and related data to a file."""
        faiss.write_index(self.index, f"{filepath}.index")
        data = {
            "history": self.history,
            "embeddings": [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in self.embeddings],
            "id_to_index": self.id_to_index
        }
        with open(f"{filepath}.json", "w") as f:
            json.dump(data, f)

    def load_index(self, filepath: str) -> None:
        """Load the FAISS index and related data from a file."""
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.json", "r") as f:
            data = json.load(f)
        self.history = data["history"]
        self.embeddings = [np.array(emb) for emb in data["embeddings"]]
        self.id_to_index = {int(k): v for k, v in data["id_to_index"].items()}

def process_query(
    query: str,
    content: Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 1
) -> Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]:
    """
    Process a query and return relevant content based on similarity.

    Args:
        query (str): The input query.
        content (Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]): The content to search through.
        embedding_model (str): The name of the embedding model to use.
        k (int): The number of top results to return.

    Returns:
        Union[str, List[str], Dict[str, List[str]], List[Dict[str, List[str]]]]: The relevant content.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    query_embedding = embeddings.embed_query(query)

    if isinstance(content, str):
        return content  # If content is a single string, return it as is

    elif isinstance(content, list) and all(isinstance(item, str) for item in content):
        # If content is a list of strings
        content_embeddings = [embeddings.embed_query(item) for item in content]
        similarities = [np.dot(query_embedding, ce) for ce in content_embeddings]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [content[i] for i in top_k_indices]

    elif isinstance(content, dict) and all(isinstance(value, list) for value in content.values()):
        # If content is a dictionary with list values
        results = {}
        for key, value_list in content.items():
            value_embeddings = [embeddings.embed_query(str(item)) for item in value_list]
            similarities = [np.dot(query_embedding, ve) for ve in value_embeddings]
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            results[key] = [value_list[i] for i in top_k_indices]
        return results

    elif isinstance(content, list) and all(isinstance(item, dict) for item in content):
        # If content is a list of dictionaries
        results = []
        for item in content:
            item_results = {}
            for key, value_list in item.items():
                if isinstance(value_list, list):
                    value_embeddings = [embeddings.embed_query(str(v)) for v in value_list]
                    similarities = [np.dot(query_embedding, ve) for ve in value_embeddings]
                    top_k_indices = np.argsort(similarities)[-k:][::-1]
                    item_results[key] = [value_list[i] for i in top_k_indices]
                else:
                    item_results[key] = value_list
            results.append(item_results)
        return results

    else:
        raise ValueError("Unsupported content type")

class HistorySession_updated:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.history: List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]] = []
        self.embeddings: List[np.ndarray] = []
        self.index = faiss.IndexFlatL2(384)  # Assuming 384-dimensional embeddings
        self.id_to_index: Dict[int, int] = {}

    def add_history(self, query: str, response: Union[str, List[str], Dict[str, List[str]]]) -> None:
        """Add a new entry to the history."""
        history_id = len(self.history)
        self.history.append({"query": query, "response": response})
        embedding = self.embedding_model.embed_query(query)
        embedding = np.array(embedding)  # Ensure embedding is a numpy array
        self.embeddings.append(embedding)
        self.index.add(np.array([embedding]))
        self.id_to_index[history_id] = len(self.embeddings) - 1

    def calculate_similarity(self, query: str) -> np.ndarray:
        """Calculate similarity between the query and all history entries."""
        query_embedding = self.embedding_model.embed_query(query)
        return np.dot(np.array(self.embeddings), query_embedding)  # Ensure self.embeddings is converted to numpy array

    def retrieve(self, query: str, k: int = 1) -> List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Retrieve the top k most similar history entries."""
        query_embedding = self.embedding_model.embed_query(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.history[self.id_to_index[i]] for i in indices[0]]

    def batch_retrieve(self, queries: List[str], k: int = 1) -> List[List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]]:
        """Batch retrieve the top k most similar history entries for multiple queries."""
        query_embeddings = [self.embedding_model.embed_query(query) for query in queries]
        distances, indices = self.index.search(np.array(query_embeddings), k)
        return [[self.history[self.id_to_index[i]] for i in query_indices] for query_indices in indices]

    def update_history(self, history_id: int, query: str, response: Union[str, List[str], Dict[str, List[str]]]) -> None:
        """Update an existing history entry."""
        if history_id < len(self.history):
            self.history[history_id] = {"query": query, "response": response}
            new_embedding = self.embedding_model.embed_query(query)
            new_embedding = np.array(new_embedding)  # Ensure new_embedding is a numpy array
            self.embeddings[self.id_to_index[history_id]] = new_embedding
            self.index.remove_ids(np.array([history_id]))
            self.index.add(np.array([new_embedding]))

    def remove_history(self, history_id: int) -> None:
        """Remove a history entry."""
        if history_id < len(self.history):
            del self.history[history_id]
            index_to_remove = self.id_to_index[history_id]
            del self.embeddings[index_to_remove]
            self.index.remove_ids(np.array([history_id]))
            del self.id_to_index[history_id]
            # Update id_to_index mapping
            for i in range(history_id + 1, len(self.history) + 1):
                if i in self.id_to_index:
                    self.id_to_index[i - 1] = self.id_to_index[i] - 1
                    del self.id_to_index[i]

    def get_history_count(self) -> int:
        """Get the total number of history entries."""
        return len(self.history)

    def get_history_by_id(self, history_id: int) -> Optional[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Get a history entry by its ID."""
        return self.history[history_id] if history_id < len(self.history) else None

    def get_history_index(self, history_id: int) -> Optional[int]:
        """Get the index of a history entry by its ID."""
        return self.id_to_index.get(history_id)

    def hybrid_search(self, query: str, k: int = 1, alpha: float = 0.5) -> List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Perform a hybrid search combining embedding similarity and keyword matching."""
        embedding_results = self.retrieve(query, k)
        keyword_results = self._keyword_search(query, k)
        
        combined_results = {}
        for i, result in enumerate(embedding_results):
            combined_results[i] = (1 - alpha) * (k - i) / k

        for i, result in enumerate(keyword_results):
            if i in combined_results:
                combined_results[i] += alpha * (k - i) / k
            else:
                combined_results[i] = alpha * (k - i) / k

        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        return [self.history[i] for i, _ in sorted_results[:k]]

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Union[str, List[str], Dict[str, List[str]]]]]:
        """Perform a simple keyword-based search."""
        query_words = set(query.lower().split())
        scores = []
        for entry in self.history:
            entry_words = set(entry["query"].lower().split())
            score = len(query_words.intersection(entry_words)) / len(query_words)
            scores.append(score)
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.history[i] for i in top_k_indices]

    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and related data to a file."""
        faiss.write_index(self.index, f"{filepath}.index")
        data = {
            "history": self.history,
            "embeddings": [emb.tolist() for emb in self.embeddings],  # Ensure tolist() is called correctly
            "id_to_index": self.id_to_index
        }
        with open(f"{filepath}.json", "w") as f:
            json.dump(data, f)

    def load_index(self, filepath: str) -> None:
        """Load the FAISS index and related data from a file."""
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.json", "r") as f:
            data = json.load(f)
        self.history = data["history"]
        self.embeddings = [np.array(emb) for emb in data["embeddings"]]  # Convert loaded embeddings back to numpy arrays
        self.id_to_index = {int(k): v for k, v in data["id_to_index"].items()}
# Example usage
if __name__ == "__main__":
    # Create a HistorySession instance
    session = HistorySession()

    # Add some history entries
    session.add_history("What is the capital of France?", "The capital of France is Paris.")
    session.add_history("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet.")
    session.add_history("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius.")

    # Retrieve similar entries
    query = "Tell me about Paris"
    results = session.retrieve(query, k=2)
    print("Retrieved results:")
    for result in results:
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print()

    # Perform a hybrid search
    hybrid_results = session.hybrid_search(query, k=2)
    print("Hybrid search results:")
    for result in hybrid_results:
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print()

    # Process a query with different content types
    string_content = "This is a single string content."
    list_content = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
    dict_content = {
        "fruits": ["Apple", "Banana", "Cherry"],
        "vegetables": ["Carrot", "Broccoli", "Spinach"]
    }
    list_dict_content = [
        {"category": "Fruits", "items": ["Apple", "Banana", "Cherry"]},
        {"category": "Vegetables", "items": ["Carrot", "Broccoli", "Spinach"]}
    ]

    print("Processing query with different content types:")
    print("String content:", process_query("Test query", string_content))
    print("List content:", process_query("Apple", list_content, k=2))
    print("Dict content:", process_query("Vegetable", dict_content, k=2))
    print("List of dict content:", process_query("Fruit", list_dict_content, k=2))

    # Save and load index
    session.save_index("history_session")
    new_session = HistorySession()
    new_session.load_index("history_session")
    print("Loaded session history count:", new_session.get_history_count())
    