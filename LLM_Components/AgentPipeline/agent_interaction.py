
import sys
from pathlib import Path
import logging
import os
import json
from typing import List, Dict, Any
from g4f.client import Client
from agentdatacollection import AgentRAGPipeline
from agenthistory import AgentHistoryManagerContentRetrieveUpdate
from langchain_huggingface import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=r'C:\Users\heman\Desktop\components\logs\testing.log',
    filemode='w'
)

HISTORY_FILE = 'history.json'
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # @param {type:"string"}
MODEL_KWARGS = {'device': 'cpu'}  # @param {type:"string"}
ENCODE_KWARGS = {'normalize_embeddings': False}  # @param {type:"string"}
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)

OUTPUT_FOLDER = r"C:\Users\heman\Desktop\components\cookbook\notebooks\en"


def load_history(file_path: str) -> List[Dict[str, Any]]:
    """Load the history from a JSON file."""
    if not os.path.exists(file_path):
        logging.info("History file not found. Creating a new one.")
        return []

    with open(file_path, 'r') as file:
        return json.load(file)


def save_history(file_path: str, history: List[Dict[str, Any]]) -> None:
    """Save the history to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(history, file, indent=4)
    logging.info(f"History successfully saved to {file_path}.")


def query_model(query: str, history: List[Dict[str, Any]]) -> str:
    """Query the model and update the history."""
    try:
        client = Client()
        prompt = f"write code in python for Query: {query} \nContent: {history}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        response_content = response.choices[0].message.content
        logging.info("Model response received successfully.")

        history_entry = {
            "query": query,
            "response": response_content
        }
        history.append(history_entry)

        return response_content

    except Exception as e:
        logging.error(f"An error occurred while querying the model: {e}")
        return "An error occurred while processing your query."


def interactive_session():
    """Start an interactive session for querying the model."""
    history = load_history(HISTORY_FILE)
    agent_history = AgentHistoryManagerContentRetrieveUpdate(EMBEDDING_MODEL)
    agent_data_pipeline = AgentRAGPipeline(
        embedding_model=EMBEDDING_MODEL,
        directory_path=OUTPUT_FOLDER
    )

    agent_data_pipeline.load_documents()
    agent_data_pipeline.split_documents()
    agent_data_pipeline.create_vectorstore()

    while True:
        try:
            query = input("Enter your query (or type 'exit' to quit): ").strip()
            if query.lower() in ['exit', 'quit']:
                break

            results = agent_data_pipeline.query(query=query, k=1)
            meta_data = [{"query": query}]
            result_session = agent_history.query(query, results, meta_data)
            results.extend(result_session[0][0].page_content)
            response = query_model(query, history=results)
            print(response)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

    save_history(HISTORY_FILE, history)


if __name__ == "__main__":
    interactive_session()
