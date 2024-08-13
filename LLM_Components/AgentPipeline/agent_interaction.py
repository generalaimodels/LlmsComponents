import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional,Tuple
from g4f.client import Client
from agentdatacollection import AgentRAGPipeline
from agenthistory import AgentHistoryManagerContentRetrieveUpdate
from langchain.docstore.document import Document
import os
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=Path('E:/LLMS/Fine-tuning/logs/testing.log'),
    filemode='w'
)
logger = logging.getLogger(__name__)

HISTORY_FILE = Path('E:/LLMS/Fine-tuning/llms-data/history.json')
OUTPUT_FOLDER = Path("E:/LLMS/Fine-tuning/AGI_papers/Doc") 

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': False}
DISTANCE_STRATEGY = DistanceStrategy.COSINE  # Choose from available options


# Initialize Embedding Model & G4F Client
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)
client = Client()

def process_folder(folder_path: str) -> None:
    """
    Processes a folder containing JSON files, extracting and appending query results.

    Args:
        folder_path (str): Path to the folder containing JSON files.
    """
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            process_file(filepath)

def process_file(filepath: str) -> None:
    """
    Processes a single JSON file, extracting and appending query results.

    Args:
        filepath (str): Path to the JSON file.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    for entry in data:
        query = entry.get("query")
        if query:
            retrieved_content = entry.get("retrieved_content", [])
            response = generate_response(query, retrieved_content)
            entry["response"] = response

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def generate_response(query: str, retrieved_content: List[str]) -> str:
    """
    Generates a response based on the query and retrieved content.

    Args:
        query (str): The user's query.
        retrieved_content (List[str]): List of potentially relevant content.

    Returns:
        str: The generated response.
    """
    relevant_content = find_relevant_content(query, retrieved_content)
    prompt = f"Query: {query}  {relevant_content} given response in great details novelity and think about query and generates properly regarding query"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def find_relevant_content(query: str, retrieved_content: List[str]) -> str:
    """
    Finds the most relevant content based on the query using FAISS.

    Args:
        query (str): The user's query.
        retrieved_content (List[str]): List of potentially relevant content.

    Returns:
        str: The most relevant content, or an empty string if none found.
    """
    if not retrieved_content:
        return ""

    # Create FAISS index
    documents = [Document(page_content=content) for content in retrieved_content]
    db = FAISS.from_documents(documents, EMBEDDING_MODEL)

    # Search for relevant content
    results = db.search(query, search_type="similarity_score_threshold",k=1)
    if results and len(results) < 2:  # 95% similarity threshold
        return results[0].page_content
    return ""

def load_history() -> List[Dict[str, str]]:
    """Load the history from a JSON file."""
    if not HISTORY_FILE.exists():
        logger.info("History file not found. Creating a new one.")
        return []

    try:
        with HISTORY_FILE.open('r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.error("Error decoding JSON from history file. Creating a new history.")
        return []


def save_history(history: List[Dict[str, str]]) -> None:
    """Save the history to a JSON file."""
    try:
        with HISTORY_FILE.open('w') as file:
            json.dump(history, file, indent=4)
        logger.info(f"History successfully saved to {HISTORY_FILE}")
    except IOError as e:
        logger.error(f"Error saving history to file: {e}")


def query_model(query: str, context: str) -> Optional[str]:
    """Query the model and return the response."""
    try:
        client = Client()
        prompt = f"Query: {query} {context} given response in great details novelity and think about query and generates properly regarding query"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"An error occurred while querying the model: {e}")
        return None


def interactive_session():
    """Start an interactive session for querying the model."""
    history = load_history()
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
            retrieved_content = results + [result[0].page_content for result in result_session]
            context=find_relevant_content(query=query, retrieved_content=retrieved_content)
            
            response = query_model(query, context=context)
            
            if response:
                history_entry = {
                    "query": query,
                    "retrieved_content": context,
                    "response": response
                }
                history.append(history_entry)
                print(response)
            else:
                print("Failed to generate a response. Please try again.")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

    save_history(history)


if __name__ == "__main__":
    interactive_session()
    
    




# import os
# import json
# import logging
# from typing import List, Dict, Any
# from pathlib import Path

# from g4f.client import Client
# from agentdatacollection import AgentRAGPipeline
# from agenthistory import AgentHistoryManagerContentRetrieveUpdate
# from langchain_huggingface import HuggingFaceEmbeddings

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename=r'C:\Users\heman\Desktop\components\logs\testing.log',
#     filemode='w'
# )

# HISTORY_FILE = 'history.json'
# logger = logging.getLogger(__name__)

# # Model Configuration
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# MODEL_KWARGS = {'device': 'cpu'}
# ENCODE_KWARGS = {'normalize_embeddings': False}
# EMBEDDING_MODEL = HuggingFaceEmbeddings(
#     model_name=MODEL_NAME,
#     model_kwargs=MODEL_KWARGS,
#     encode_kwargs=ENCODE_KWARGS
# )

# OUTPUT_FOLDER = r"E:\LLMS\Fine-tuning\AGI_papers\Doc"


# def load_history(file_path: str) -> List[Dict[str, str]]:
#     """Load the history from a JSON file."""
#     if not os.path.exists(file_path):
#         logger.info("History file not found. Creating a new one.")
#         return []

#     with open(file_path, 'r') as file:
#         return json.load(file)


# def save_history(file_path: str, history: List[Dict[str, str]]) -> None:
#     """Save the history to a JSON file."""
#     with open(file_path, 'w') as file:
#         json.dump(history, file, indent=4)
#     logger.info(f"History successfully saved to {file_path}.")


# def query_model(query: str, retrieved_content: str) -> str:
#     """Query the model and get the response."""
#     try:
#         client = Client()
#         prompt = f"Write code in Python for Query: {query} \nContent: {retrieved_content}"
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         response_content = response.choices[0].message.content
#         logger.info("Model response received successfully.")
#         return response_content

#     except Exception as e:
#         logger.error(f"An error occurred while querying the model: {e}")
#         return "An error occurred while processing your query."


# def interactive_session() -> None:
#     """Start an interactive session for querying the model."""
#     history = load_history(HISTORY_FILE)
#     agent_history = AgentHistoryManagerContentRetrieveUpdate(EMBEDDING_MODEL)
#     agent_data_pipeline = AgentRAGPipeline(
#         embedding_model=EMBEDDING_MODEL,
#         directory_path=OUTPUT_FOLDER
#     )

#     agent_data_pipeline.load_documents()
#     agent_data_pipeline.split_documents()
#     agent_data_pipeline.create_vectorstore()

#     while True:
#         try:
#             query = input("Enter your query (or type 'exit' to quit): ").strip()
#             if query.lower() in ['exit', 'quit']:
#                 break

#             results = agent_data_pipeline.query(query=query, k=1)
#             retrieved_content = "\n".join(result.page_content for result in results)
#             response = query_model(query, retrieved_content)

#             history_entry = {
#                 "query": query,
#                 "response": response,
#                 "retrieved_content": retrieved_content
#             }
#             history.append(history_entry)
#             print(response)

#         except KeyboardInterrupt:
#             print("\nSession interrupted. Exiting.")
#             break

#         except Exception as e:
#             logger.error(f"An unexpected error occurred: {e}")
#             print("An error occurred. Please try again.")

#     save_history(HISTORY_FILE, history)


# if __name__ == "__main__":
#     interactive_session()