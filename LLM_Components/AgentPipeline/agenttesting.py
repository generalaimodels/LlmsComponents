# import sys
# from pathlib import Path
# import logging
# import json
# import os
# from typing import List, Dict, Any, Optional

# from g4f.client import Client
# from langchain_huggingface import HuggingFaceEmbeddings
# from agentdatacollection import AgentRAGPipeline
# from agenthistory import AgentHistoryManagerContentRetrieveUpdate

# # Add parent directory to sys.path
# sys.path.append(str(Path(__file__).resolve().parent))

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='logs/testing_agent.log',
#     filemode='w'
# )
# logger = logging.getLogger(__name__)

# # Constants
# HISTORY_FILE = 'history.json'
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# MODEL_KWARGS = {'device': 'cpu'}
# ENCODE_KWARGS = {'normalize_embeddings': False}
# OUTPUT_FOLDER = Path(r"C:\Users\heman\Desktop\components\LlmsComponents\Agents")


# def load_history(file_path: str) -> List[Dict[str, Any]]:
#     """Load the history from a JSON file."""
#     try:
#         with open(file_path, 'r') as file:
#             return json.load(file)
#     except FileNotFoundError:
#         logger.info("History file not found. Creating a new one.")
#         return []
#     except json.JSONDecodeError:
#         logger.error("Error decoding JSON from history file.")
#         return []


# def save_history(file_path: str, history: List[Dict[str, Any]]) -> None:
#     """Save the history to a JSON file."""
#     try:
#         with open(file_path, 'w') as file:
#             json.dump(history, file, indent=4)
#         logger.info(f"History successfully saved to {file_path}.")
#     except IOError:
#         logger.error(f"Error saving history to {file_path}.")


# def query_model(query: str, history: List[Dict[str, Any]]) -> str:
#     """Query the model and update the history."""
#     try:
#         client = Client()
#         prompt = f"write code in python for Query: {query} content: {history}"
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#         )
#         response_content = response.choices[0].message.content
#         logger.info("Model response received successfully.")

#         history_entry = {
#             "query": query,
#             "response": response_content
#         }
#         history.append(history_entry)

#         return response_content

#     except Exception as e:
#         logger.error(f"An error occurred while querying the model: {e}")
#         return "An error occurred while processing your query."


# def create_embedding_model() -> HuggingFaceEmbeddings:
#     """Create and return the embedding model."""
#     return HuggingFaceEmbeddings(
#         model_name=MODEL_NAME,
#         model_kwargs=MODEL_KWARGS,
#         encode_kwargs=ENCODE_KWARGS
#     )


# def interactive_session(
#     embedding_model: HuggingFaceEmbeddings,
#     agent_data_pipeline: AgentRAGPipeline,
#     agent_history: AgentHistoryManagerContentRetrieveUpdate
# ) -> None:
#     """Start an interactive session for querying the model."""
#     history = load_history(HISTORY_FILE)

#     while True:
#         try:
#             query = input("Enter your query (or type 'exit' to quit): ").strip()
#             if query.lower() in ['exit', 'quit']:
#                 break
            
#             results = agent_data_pipeline.query(query=query, k=1)
#             meta_data = [{"query": query}]
#             result_session = agent_history.query(query, results, meta_data)
#             results.extend(result_session[0][0].page_content)
#             response = query_model(query, history=results)
#             print(response)

#         except KeyboardInterrupt:
#             print("\nSession interrupted. Exiting.")
#             break

#     save_history(HISTORY_FILE, history)


# def main() -> None:
#     """Main function to set up and run the interactive session."""
#     logger.info("Agent is starting...")
    
#     embedding_model = create_embedding_model()
    
#     agent_history = AgentHistoryManagerContentRetrieveUpdate(embedding_model)
    
#     agent_data_pipeline = AgentRAGPipeline(
#         embedding_model=embedding_model,
#         directory_path=OUTPUT_FOLDER,
#         exclude_patterns=["*.pyc"],
       

#     )

#     agent_data_pipeline.load_documents()
#     agent_data_pipeline.split_documents()
#     agent_data_pipeline.create_vectorstore()

#     interactive_session(embedding_model, agent_data_pipeline, agent_history)

#     logger.info("Agent session ended.")


# if __name__ == "__main__":
#     main()



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









# ConvertFolder_to_TxtFolder(
#     input_folders=input_folder,
#     output_folder=output_folder
# )
# logger.info(f"Agent input folder {input_folder} outputdir {output_folder}")
# agentdataset=AgentDataset(r"E:\LLMS\Fine-tuning\output").load_data()
# files_names=list(agentdataset.keys())
# logger.info(f"Only this files are selected for Agent to understanding content {files_names}")
# agentdata=AgentDatasetLoader(dataset=agentdataset)
# page_content_multi, metadata_multi=agentdata.load_multiple_datasets(dataset_names=files_names)
# Agent_retriever=AgentContentRetrieval(embedding_model=embedding_model)
# Agent_history=AgentHistoryManagerContentRetrieveUpdate(Agent_retriever.embeddings)
# Agent_retriever.add_documents(page_content_multi, metadata_multi)

# Agentdataragpipeline=AgentRAGPipeline(
#     embedding_model=embedding_model,
#     directory_path=output_folder)

# Agentdataragpipeline.load_documents()
# Agentdataragpipeline.split_documents()
# Agentdataragpipeline.create_vectorstore()

# # Agentdataragpipeline.save_vectorstore(output_folder)
# query = "Exaplin great detail about AgentDatasetloader"
# # # Later, you can load the vectorstore and query it
# # Agentdataragpipeline.load_vectorstore(output_folder)
# results =Agentdataragpipeline.query(query=query, k=5)
# print(results)


# client = Client()
# prompt=f" write code in python for  Query: {query} content:{results}"
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": prompt}],
   
# )
# print(response.choices[0].message.content)
# # # # Test single retrieval

# results = Agent_retriever.retrieve(query, top_k=4)
# print(f"\nQuery: {query}")
# for doc, score in results:
#     print(f"Document: {doc.page_content}")
#     print(f"Score: {score}")
#     print(f"Metadata: {doc.metadata}")
#     print()


# query = "Exaplin great detail about AgentDatasetloader"
# results = Agent_history.query(query, page_content_multi, metadata_multi )

# for doc, score in results:
#     print(f"Document: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")
#     print(f"Similarity Score: {score}")
#     print("---")


# print(response.choices[0].message.content)
# meta_data_llms={"rosponser":response.choices[0].message.content}
# Agent_history.update_documents(new_documents=[response.choices[0].message.content],new_metadata=meta_data_llms)
# query = "write python code for  AgentDatasetloader"
# results = Agent_history.query(query, page_content_multi, metadata_multi )

# for doc, score in results:
#     print(f"Document: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")
#     print(f"Similarity Score: {score}")
#     print("---")

# Agentdataragpipeline

# prompt=f" write code in python for  Query: {query} content:{results[0][0].page_content} metadata {results[0][0].metadata}"
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": prompt}],
   
# )
# meta_data_llms={"rosponser":response.choices[0].message.content}
# Agent_history.update_documents(new_documents=[response.choices[0].message.content],new_metadata=meta_data_llms)
# results = Agent_history.query(query, page_content_multi, metadata_multi )


# results = Agent_history.query(query, page_content_multi, metadata_multi )
# prompt=f" write code in python for  Query: {query} content:{results[0][0].page_content} metadata {results[0][0].metadata}"
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": prompt}],
   
# )
# print(response.choices[0].message.content)


