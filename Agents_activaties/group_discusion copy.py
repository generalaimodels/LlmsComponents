import asyncio
import json
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from g4f import ChatCompletion

# Define typing hints for clarity
JSONType = Dict[str, Any]
QueryResponsePair = Tuple[str, str]

class KnowledgeBase:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self._load_knowledge_base()

    def _load_knowledge_base(self) -> JSONType:
        if Path(self.file_path).exists():
            with open(self.file_path, 'r', encoding='utf-8') as json_file:
                return json.load(json_file)
        return {}

    def query(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def update(self, key: str, value: str) -> None:
        self.data[key] = value
        self._save_knowledge_base()

    def _save_knowledge_base(self) -> None:
        with open(self.file_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file, indent=4)


class HistoryManager:
    def __init__(self) -> None:
        self.history: List[QueryResponsePair] = []

    def add_to_history(self, query: str, response: str) -> None:
        self.history.append((query, response))

    def get_history(self) -> List[QueryResponsePair]:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()


class IntentEntityRecognizer:
    # A simplified placeholder for intent and entity recognition.
    def recognize(self, query: str) -> Tuple[str, List[str]]:
        if "capital" in query:
            return "GetCapital", []
        elif "how" in query:
            return "ExplainProcess", []
        elif "theory" in query:
            return "ExplainTheory", []
        return "GeneralQuery", []


class ResponseGenerator:
    @staticmethod
    async def generate_response(query: str) -> str:
        response = await asyncio.to_thread(
            ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
        )
        return response


class DialogueManager:
    def __init__(self, knowledge_base: KnowledgeBase, history_manager: HistoryManager) -> None:
        self.knowledge_base = knowledge_base
        self.history_manager = history_manager
        self.intent_recognizer = IntentEntityRecognizer()

    async def handle_query(self, query: str) -> str:
        intent, entities = self.intent_recognizer.recognize(query)
        response = self.knowledge_base.query(query)

        if not response:
            response = await ResponseGenerator.generate_response(query)
            self.knowledge_base.update(query, response)

        self.history_manager.add_to_history(query, response)
        return response


async def main() -> None:
    knowledge_base = KnowledgeBase(file_path="knowledge_base.json")
    history_manager = HistoryManager()
    dialogue_manager = DialogueManager(knowledge_base, history_manager)

    queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Explain the theory of relativity.",
        # Additional queries can be added here
    ]

    tasks = [dialogue_manager.handle_query(query) for query in queries]
    results = await asyncio.gather(*tasks)

    for query, result in zip(queries, results):
        print(f"Query: {query}")
        print(f"Response: {result}")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())