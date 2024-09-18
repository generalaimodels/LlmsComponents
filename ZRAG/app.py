import os
import gradio as gr
from typing import List, Tuple, Optional
from ragsystem import create_knowledge_vector_database
from llmengine import get_llm_response
from raglogger import setup_logger
from functools import lru_cache

# Initialize logger
logger = setup_logger()

# Caching previously created knowledge vector databases
@lru_cache(maxsize=5)
def create_cached_knowledge_vector_database(dataset_name: str):
    """
    Cache the knowledge vector database to reduce redundant creation.
    This prevents the re-creation of the vector database for datasets already processed.
    
    Args:
        dataset_name (str): The dataset name to create the vector database for.

    Returns:
        The knowledge vector database object if successful, else None.
    """
    knowledge_vector_database, _ = create_knowledge_vector_database(dataset_name)
    if knowledge_vector_database:
        logger.info(f"Cached Knowledge Vector Database for '{dataset_name}' created successfully.")
        return knowledge_vector_database
    else:
        logger.error(f"Failed to create the Knowledge Vector Database for '{dataset_name}'.")
        return None

def retrieve_documents(user_query: str, dataset_name: str = "fka/awesome-chatgpt-prompts", top_k: int = 3) -> Tuple[str, List[str]]:
    """
    Retrieves top K documents based on the user's query using similarity search on a knowledge vector database.
    
    Args:
        user_query (str): The query input by the user.
        dataset_name (str, optional): The dataset to use for vector search. Defaults to "fka/awesome-chatgpt-prompts".
        top_k (int, optional): The number of top documents to return. Defaults to 10.

    Returns:
        Tuple[str, List[str]]: A tuple containing the concatenated context from retrieved docs and individual doc contents.
                               If no documents are found or an error occurs, returns an empty string and list.
    """
    try:
        # Create or retrieve cached knowledge vector database
        knowledge_vector_database = create_cached_knowledge_vector_database(dataset_name)
        if knowledge_vector_database is None:
            return "", []

        # Retrieve documents (keeping it single-threaded to avoid "daemonic" process error)
        logger.info(f"Starting document retrieval for query: '{user_query}' using dataset '{dataset_name}'")
        retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=top_k)
        
        if not retrieved_docs:
            logger.warning("No documents were retrieved for the query.")
            return "", []

        # Extract page contents from the retrieved documents
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]

        # Creating a contextual summary from the retrieved documents
        context = "\n".join([f"Document {i + 1}: {text}" for i, text in enumerate(retrieved_docs_text)])

        logger.info(f"Successfully retrieved {len(retrieved_docs)} documents.")
        return context, retrieved_docs_text

    except Exception as e:
        logger.error(f"Error during document retrieval: {str(e)}")
        return "", []

def process_query(user_query: str, dataset_name: str = "fka/awesome-chatgpt-prompts", top_k: int = 3) -> Tuple[str, str]:
    """
    Process the user query by retrieving documents from the dataset and generating a response via an LLM.
    
    Args:
        user_query (str): The query input by the user.
        dataset_name (str, optional): The dataset to use for vector search. Defaults to "fka/awesome-chatgpt-prompts".
        top_k (int, optional): The number of top retrieved documents. Defaults to 10.
    
    Returns:
        Tuple[str, str]: The retrieved document context and the LLM's response.
                         If no documents found, appropriate messages are returned.
    """
    if not user_query:
        return "Query cannot be empty.", ""

    logger.info(f"Processing query: '{user_query}' with dataset '{dataset_name}'")

    # Retrieve documents from the knowledge vector database
    context, docs = retrieve_documents(user_query, dataset_name, top_k)

    if docs:
        # Using few-shot prompting technique for generating better responses
        prompt = f"Use the following retrieved documents to answer the query: {user_query}\n\nContext:\n{context}\nPlease provide a concise and relevant response."

        # Define the model name (considering future expansion or more complex models)
        model_name = "gpt-4o-mini"

        # Get response from the LLM
        llm_response = get_llm_response(model_name, prompt)

        logger.info(f"LLM response generated successfully for query: '{user_query}'")
        return context, llm_response

    logger.warning(f"No documents found for query: '{user_query}'")
    return "No documents found for the input query.", ""

def main_interface():
    """
    Main function to define and launch the user interface using Gradio.
    """
    interface = gr.Interface(
        fn=lambda query: process_query(query),
        inputs=gr.Textbox(label="Enter your query"),
        outputs=[
            gr.Textbox(label="Retrieved Document Context"),
            gr.Textbox(label="LLM Response")
        ],
        title="Document Search and LLM Query System",
        description="Enter a query to retrieve relevant documents from a dataset. The LLM generates a response based on the retrieved documents.",
        live=False  # Set live=False to avoid excessive requests during typing
    )

    interface.launch(share=True)

if __name__ == "__main__":
    main_interface()