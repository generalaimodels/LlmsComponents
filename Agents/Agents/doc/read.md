```python
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import util
import faiss
import json
import os

class LLMsReader:
    def __init__(self, embedding_model: str, device: str = "cuda"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model, device=device)
        self.history: List[Dict[str, Union[str, List[str]]]] = []
        self.index = None
        self.documents: List[Document] = []

    def add_history(self, query: str, response: Union[str, List[str], Dict[str, List[str]]]) -> None:
        self.history.append({"query": query, "response": response})

    def calculate_similarity(self, query_embedding: np.ndarray, content_embeddings: np.ndarray) -> np.ndarray:
        return util.cos_sim(query_embedding, content_embeddings)[0]

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        query_embedding = self.embedding_model.embed_query(query)
        if self.index is None:
            self._build_index()
        
        D, I = self.index.search(np.array([query_embedding]), k)
        return [(self.documents[i], score) for i, score in zip(I[0], D[0])]

    def batch_retrieve(self, queries: List[str], k: int = 5) -> List[List[Tuple[Document, float]]]:
        query_embeddings = self.embedding_model.embed_documents(queries)
        if self.index is None:
            self._build_index()
        
        D, I = self.index.search(np.array(query_embeddings), k)
        return [[(self.documents[i], score) for i, score in zip(row_I, row_D)] for row_I, row_D in zip(I, D)]

    def update_history(self, index: int, new_response: Union[str, List[str], Dict[str, List[str]]]) -> None:
        if 0 <= index < len(self.history):
            self.history[index]["response"] = new_response
        else:
            raise IndexError("History index out of range")

    def remove_history(self, index: int) -> None:
        if 0 <= index < len(self.history):
            del self.history[index]
        else:
            raise IndexError("History index out of range")

    def get_history_count(self) -> int:
        return len(self.history)

    def get_history_by_id(self, index: int) -> Optional[Dict[str, Union[str, List[str]]]]:
        if 0 <= index < len(self.history):
            return self.history[index]
        return None

    def get_history_index(self, query: str) -> Optional[int]:
        for i, item in enumerate(self.history):
            if item["query"] == query:
                return i
        return None

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        semantic_results = self.retrieve(query, k)
        keyword_results = self._keyword_search(query, k)
        
        combined_results = {}
        for doc, score in semantic_results:
            combined_results[doc] = alpha * score
        
        for doc, score in keyword_results:
            if doc in combined_results:
                combined_results[doc] += (1 - alpha) * score
            else:
                combined_results[doc] = (1 - alpha) * score
        
        return sorted([(doc, score) for doc, score in combined_results.items()], key=lambda x: x[1], reverse=True)[:k]

    def save_index(self, path: str) -> None:
        if self.index is not None:
            faiss.write_index(self.index, f"{path}/faiss_index.bin")
            with open(f"{path}/documents.json", "w") as f:
                json.dump([doc.to_dict() for doc in self.documents], f)

    def load_index(self, path: str) -> None:
        if os.path.exists(f"{path}/faiss_index.bin") and os.path.exists(f"{path}/documents.json"):
            self.index = faiss.read_index(f"{path}/faiss_index.bin")
            with open(f"{path}/documents.json", "r") as f:
                self.documents = [Document.from_dict(doc) for doc in json.load(f)]
        else:
            raise FileNotFoundError("Index files not found")

    def _build_index(self) -> None:
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in self.documents])
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings))

    def _keyword_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        query_terms = set(query.lower().split())
        scores = []
        for doc in self.documents:
            doc_terms = set(doc.page_content.lower().split())
            score = len(query_terms.intersection(doc_terms)) / len(query_terms)
            scores.append((doc, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def interactive_mode(self) -> None:
        print("Welcome to Interactive Mode. Type 'exit' to quit.")
        while True:
            query = input("Enter your query: ")
            if query.lower() == 'exit':
                break
            
            response = self.process_query(query)
            print("Response:", response)
            
            self.add_history(query, response)

    def process_query(self, query: str) -> Union[str, List[str], Dict[str, List[str]]]:
        results = self.hybrid_search(query)
        response = [doc.page_content for doc, _ in results]
        return response

if __name__ == "__main__":
    reader = LLMsReader(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    # Add some sample documents
    reader.documents = [
        Document(page_content="The quick brown fox jumps over the lazy dog"),
        Document(page_content="A journey of a thousand miles begins with a single step"),
        Document(page_content="To be or not to be, that is the question"),
    ]
    reader.interactive_mode()

```
------

```python
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import torch
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class LLMsReader:
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device=device)
        self.embeddings = HuggingFaceEmbeddings(model_name)
        self.history = []

    def add_history(self, query: str, response: Union[str, List[str], Dict[str, Union[List[str], List[Dict[str, List[str]]]]]]) -> None:
        self.history.append({'query': query, 'response': response})

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def retrieve(self, query: str) -> Union[str, List[str], Dict[str, Union[List[str], List[Dict[str, List[str]]]]]]:
        query_embedding = self.embeddings.embed(query)
        most_similar_content = None
        highest_similarity = -1

        for entry in self.history:
            response_embedding = self.embeddings.embed(entry['response'])
            similarity = self.calculate_similarity(query_embedding, response_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_content = entry['response']

        return most_similar_content

    def batch_retrieve(self, queries: List[str]) -> List[Union[str, List[str], Dict[str, Union[List[str], List[Dict[str, List[str]]]]]]]:
        return [self.retrieve(query) for query in queries]

    def update_history(self, index: int, query: Optional[str] = None, response: Optional[Union[str, List[str], Dict[str, Union[List[str], List[Dict[str, List[str]]]]]]] = None) -> None:
        if index < 0 or index >= len(self.history):
            raise IndexError("Index out of range.")
        if query:
            self.history[index]['query'] = query
        if response:
            self.history[index]['response'] = response

    def remove_history(self, index: int) -> None:
        if index < 0 or index >= len(self.history):
            raise IndexError("Index out of range.")
        self.history.pop(index)

    def get_history_count(self) -> int:
        return len(self.history)

    def get_history_item(self, index: int) -> Dict[str, Union[str, Union[str, List[str], Dict[str, Union[List[str], List[Dict[str, List[str]]]]]]]]:
        if index < 0 or index >= len(self.history):
            raise IndexError("Index out of range.")
        return self.history[index]

    def hybrid_search(self, query: str) -> Union[str, List[str], Dict[str, Union[List[str], List[Dict[str, List[str]]]]]]:
        return self.retrieve(query)  # Placeholder for an actual hybrid search implementation

    def save_index(self, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            torch.save(self.history, f)

    def load_index(self, file_path: str) -> None:
        with open(file_path, 'rb') as f:
            self.history = torch.load(f)

# Example usage
if __name__ == "__main__":
    llm_reader = LLMsReader("gpt2")
    llm_reader.add_history("What is AI?", "Artificial Intelligence is a field of computer science.")
    response = llm_reader.retrieve("Tell me about AI")
    print(response)

```