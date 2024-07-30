import sys
from pathlib import Path 
import logging 
from g4f.client import Client
sys.path.append(str(Path(__file__).resolve().parents[0]))

import json
import os
from typing import List, Dict, Any
import argparse
import logging
import readline
from agentdatacollection import (
    ConvertFolder_to_TxtFolder,
    AgentDataset,
    AgentDatasetLoader,
    AgentRAGPipeline
)
from agentdataretrieval import AgentContentRetrieval
from agenthistory import AgentHistoryManagerContentRetrieveUpdate
from langchain_huggingface import HuggingFaceEmbeddings
from agentcustommodel import (
    AgentModel,
    AgentPipeline,
    AgentPreProcessorPipeline,
    BitsAndBytesConfig,
    set_seed,
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/testing_agent.log',
    filemode='w'
)

HISTORY_FILE = 'history.json'
logger = logging.getLogger(__name__)




# Model_Name=""
# cache_dir=""
# model=AgentModel.load_model(
#     model_type="causal_lm",
#     model_name_or_path=Model_Name,
#     cache_dir=cache_dir,
# )
# tokeiner=AgentPreProcessorPipeline(model_type="text",
#                                    pretrained_model_name_or_path=Model_Name,
#                                    cache_dir=cache_dir
#                                    ).process_data()
model_name = "sentence-transformers/all-mpnet-base-v2" #@param {type:"string"} 
model_kwargs = {'device': 'cpu'}  #@param {type:"string"}
encode_kwargs = {'normalize_embeddings': False} #@param {type:"string"}
embedding_model= HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
logger.info(f"Agent is Started ...")
input_folder :str =[r"E:\LLMS\Fine-tuning\LlmsComponents\LLM_Components\AgentPipeline"]
output_folder :str =r"E:\LLMS\Fine-tuning\output"










def load_history(file_path: str) -> List[Dict[str, Any]]:
    """Loads the history from a JSON file."""
    if not os.path.exists(file_path):
        logging.info("History file not found. Creating a new one.")
        return []

    with open(file_path, 'r') as file:
        return json.load(file)


def save_history(file_path: str, history: List[Dict[str, Any]]) -> None:
    """Saves the history to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(history, file, indent=4)
    logging.info(f"History successfully saved to {file_path}.")


def query_model(query: str, history: List[Dict[str, Any]]) -> str:
    """Queries the model and updates the history."""
    try:
        client = Client()
        prompt = f"write code in python for Query: {query} content: {history}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
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
    """Starts an interactive session for querying the model."""
    history = load_history(HISTORY_FILE)
    Agent_history=AgentHistoryManagerContentRetrieveUpdate(embedding_model)
    agent_data_pipeline = AgentRAGPipeline(
        embedding_model=embedding_model,  # Replace with actual embedding model path
        directory_path=output_folder  # Replace with actual directory path
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
            meta_data=[
                {"query":query}
            ]
            result_session = Agent_history.query(query, results, meta_data)
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


