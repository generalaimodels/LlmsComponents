import argparse
import json
import os
from typing import List, Dict, Any
import logging
from datetime import datetime
from agentdatacollection import  AgentRAGPipeline  # Ensure these modules are correctly imported
from g4f.client import Client
import json
import os
from typing import List, Dict, Any
import argparse
from datetime import datetime

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

class AgentRAGPipeline:
    def __init__(self, embedding_model: HuggingFaceBgeEmbeddings, directory_path: str):
        self.embedding_model = embedding_model
        self.directory_path = directory_path
        self.vectorstore = None
        self.documents = []

    def load_documents(self) -> None:
        # Implementation for loading documents
        pass

    def split_documents(self) -> None:
        # Implementation for splitting documents
        pass

    def create_vectorstore(self) -> None:
        # Implementation for creating vectorstore
        pass

    def save_vectorstore(self, output_folder: str) -> None:
        # Implementation for saving vectorstore
        pass

    def load_vectorstore(self, output_folder: str) -> None:
        # Implementation for loading vectorstore
        pass

    def query(self, query: str, k: int = 5) -> List[str]:
        # Implementation for querying vectorstore
        return ["Result 1", "Result 2", "Result 3"]  # Placeholder

class HistoryManager:
    def __init__(self, history_file: str):
        self.history_file = history_file
        self.history: List[Dict[str, Any]] = []
        self.load_history()

    def load_history(self) -> None:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading history file: {self.history_file}")
            self.history = []

    def save_history(self) -> None:
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add_entry(self, query: str, results: List[str], response: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results": results,
            "response": response
        }
        self.history.append(entry)
        self.save_history()

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

def main() -> None:
    parser = argparse.ArgumentParser(description="Command-line interface for AI agent")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--history", action="store_true", help="Show command history")
    args = parser.parse_args()

    history_manager = HistoryManager("command_history.json")

    if args.history:
        print(json.dumps(history_manager.get_history(), indent=2))
        return

    if not args.query:
        print("Please provide a query using the --query argument.")
        return

    try:
        # Initialize necessary components
        embedding_model = OpenAIEmbeddings()
        output_folder = "output"
        agent_rag_pipeline = AgentRAGPipeline(embedding_model, output_folder)

        # Process the query
        agent_rag_pipeline.load_documents()
        agent_rag_pipeline.split_documents()
        agent_rag_pipeline.create_vectorstore()

        results = agent_rag_pipeline.query(query=args.query, k=5)

        # Generate response using OpenAI
        client = Client()
        prompt = f"Write code in Python for Query: {args.query} content: {results}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        ai_response = response.choices[0].message.content

        # Print the response
        print(ai_response)

        # Save to history
        history_manager.add_entry(args.query, results, ai_response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()




def save_history(history_data: List[Dict[str, Any]], file_path: str) -> None:
    """Save the history of executions to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        logging.info(f"History successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving history: {e}")

def load_history(file_path: str) -> List[Dict[str, Any]]:
    """Load history of executions from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                history_data = json.load(f)
                logging.info(f"History successfully loaded from {file_path}")
                return history_data
        except Exception as e:
            logging.error(f"Error loading history: {e}")
            return []
    else:
        logging.warning(f"History file {file_path} not found. Starting with empty history.")
        return []

def run_query(query: str, history_file: str) -> None:
    """Run the query using the model and log history."""
    history_data = load_history(history_file)

    try:
        client = Client()
        prompt = f"write code in python for Query: {query} content:{results}"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.choices[0].message.content
        print(response_text)

        # Logging history
        history_entry = {
            "query": query,
            "response": response_text,
            "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        history_data.append(history_entry)
        save_history(history_data, history_file)

    except Exception as e:
        logging.error(f"Error during running query: {e}")

def run_pipeline(query: str, output_folder: str, history_file: str) -> None:
    """Run the RAG pipeline and log results."""
    history_data = load_history(history_file)

    try:
        agent_pipeline = AgentRAGPipeline(
            embedding_model=embedding_model,
            directory_path=output_folder
        )
        
        agent_pipeline.load_documents()
        agent_pipeline.split_documents()
        agent_pipeline.create_vectorstore()
        
        results = agent_pipeline.query(query=query, k=5)
        print(results)

        # Logging history
        history_entry = {
            "query": query,
            "results": results,
            "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        history_data.append(history_entry)
        save_history(history_data, history_file)

    except Exception as e:
        logging.error(f"Error during pipeline execution: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run queries and save the history.')
    parser.add_argument('query', type=str, help='The query to run through the model.')
    parser.add_argument('--output_folder', type=str, default='output', help='The folder to save outputs.')
    parser.add_argument('--history_file', type=str, default='history.json', help='The file to store the history of queries.')

    args = parser.parse_args()

    run_pipeline(args.query, args.output_folder, args.history_file)
    run_query(args.query, args.history_file)

if __name__ == '__main__':
    main()