import os
import sys
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
import faiss
from pathlib import Path
from transformers import AutoTokenizer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from datacollection import AdvancedDirectoryLoader
from document_splitter import AdvancedDocumentSplitter
from embedding_data import AdvancedFAISS
from g4f.client import Client
import gradio as gr


# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CONFIG = {
    "EMBEDDING_MODEL_NAME": "thenlper/gte-large",
    "DATA_DIR": r"C:\Users\heman\Desktop\Coding\LlmsComponents\LLM_Components\chat_bot",
    "CHUNK_SIZE": 512,
    "K": 5,  # Number of similar documents to retrieve
    "LLM_MODEL": "gpt-4o-mini",
}

client = Client()


def process_document(
    doc: str, 
    splitter: AdvancedDocumentSplitter,
) -> List[Dict[str, Any]]:
    """
    Process a single document using the document splitter.
    
    Args:
        doc (str): The document to be processed.
        splitter (AdvancedDocumentSplitter): The document splitter.
    
    Returns:
        List of dictionaries containing processed documents.
    """
    return splitter.split_documents([doc])


def build_vector_database(
    data_dir: str, 
    embedding_model_name: str, 
    chunk_size: int, 
) -> Optional[AdvancedFAISS]:
    """
    Build a vector database from documents in the data directory.
    
    Args:
        data_dir (str): The directory path where documents are stored.
        embedding_model_name (str): The name of the embedding model to be used.
        chunk_size (int): The chunk size for splitting documents.
        k (int): Number of similar documents to retrieve.
    
    Returns:
        AdvancedFAISS object or None: Built vector database (FAISS index) or None in case of failure.
    """
    try:
        loader = AdvancedDirectoryLoader(
            data_dir, exclude=[".pyc", "__pycache__"], silent_errors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        splitter = AdvancedDocumentSplitter(tokenizer=tokenizer, chunk_size=chunk_size)

        documents = loader.load()

        # Process documents sequentially
        docs_processed = [
            doc 
            for sublist in [process_document(doc, splitter) for doc in documents] 
            for doc in sublist
        ]

        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        sample_embedding = embeddings_model.embed_query("Sample text")
        dimension = len(sample_embedding)
        index = faiss.IndexFlatL2(dimension)
        advanced_faiss = AdvancedFAISS(
            embedding_function=embeddings_model.embed_query,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        faiss_index = advanced_faiss.from_documents(
            docs_processed, embeddings_model, distance_strategy=DistanceStrategy.COSINE
        )

        return faiss_index

    except Exception as e:
        print(f"Error building vector database: {e}")
        return None


def format_prompt(
    retrieved_docs: List[Dict[str, Any]], 
    question: str
) -> str:
    """
    Format the prompt based on retrieved documents and the user question.
    
    Args:
        retrieved_docs (List[Dict[str, Any]]): Retrieved documents from the vector database.
        question (str): The user's query.
    
    Returns:
        str: A JSON-formatted prompt for the language model.
    """
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n" + "".join(
        [f"Document {str(i)}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)]
    )

    prompt_template = [
        {
            "role": "system",
            "content": """Using the information contained in the context, 
give a comprehensive answer to the question. 
Respond only to the question asked; response should be concise and relevant to the question. 
Provide the number of the source document when relevant. 
If the answer cannot be deduced from the context, Always try to give the best of your ability."""
        },
        {
            "role": "user",
            "content": f"""Context:
{context}
---
Now here is the question you need to answer.

Question: {question}"""
        }
    ]

    return json.dumps(prompt_template, indent=2)
config=CONFIG
# Build the vector database
vector_db = build_vector_database(
    config["DATA_DIR"], config["EMBEDDING_MODEL_NAME"], config["CHUNK_SIZE"], config["K"]
)

def main(user_query: str, config: Dict[str, Any]) -> None:
    """
    Main function to orchestrate the process.
    
    Args:
        user_query (str): The query provided by the user.
        config (Dict[str, Any]): Configuration dictionary.
    """
    try:
        start_time = time.time()

        # Build the vector database
        vector_db = build_vector_database(
            config["DATA_DIR"], config["EMBEDDING_MODEL_NAME"], config["CHUNK_SIZE"], config["K"]
        )

        if vector_db:
            # Perform similarity search
            retrieved_docs = vector_db.similarity_search(user_query, k=config["K"])

            if retrieved_docs:
                # Format and print the prompt
                final_prompt = format_prompt(retrieved_docs, user_query)
                print("Generated Prompt:\n", final_prompt, "\n\n")

                # Get LLM response
                response = client.chat.completions.create(
                    model=config["LLM_MODEL"],
                    messages=[{"role": "user", "content": final_prompt}],
                )
                print("Answer:\n\n", response.choices[0].message.content)

                # Rethinker - Optional step for refining the answer
                rethinker = f"Answer to this best of your ability: {response.choices[0].message.content}"
                response = client.chat.completions.create(
                    model=config["LLM_MODEL"],
                    messages=[{"role": "user", "content": rethinker}],
                )
                print("Rethinker Answer:\n\n", response.choices[0].message.content)

            else:
                print("No documents retrieved for the query.")
        else:
            print("Failed to build vector database.")

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time} seconds")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def query_processor(user_query: str, config: Dict[str, Any]) -> str:
    """
    Process the user query and return the response.
    
    Args:
        user_query (str): The query provided by the user.
        config (Dict[str, Any]): Configuration dictionary.
    
    Returns:
        str: A string containing the initial and refined answers, as well as the execution time.
    """
    try:
        start_time = time.time()

        # Build the vector database
        vector_db = build_vector_database(
            config["DATA_DIR"], config["EMBEDDING_MODEL_NAME"], config["CHUNK_SIZE"], config["K"]
        )

        if vector_db:
            # Perform similarity search
            retrieved_docs = vector_db.similarity_search(user_query, k=config["K"])

            if retrieved_docs:
                # Format the prompt
                final_prompt = format_prompt(retrieved_docs, user_query)

                # Get LLM response
                response = client.chat.completions.create(
                    model=config["LLM_MODEL"],
                    messages=[{"role": "user", "content": final_prompt}],
                )
                answer = response.choices[0].message.content

                # Rethinker - Optional step for refining the answer
                rethinker = f"Answer to this best of your ability: {answer}"
                response = client.chat.completions.create(
                    model=config["LLM_MODEL"],
                    messages=[{"role": "user", "content": rethinker}],
                )
                refined_answer = response.choices[0].message.content

                end_time = time.time()
                execution_time = end_time - start_time

                return (
                    f"Initial Answer:\n\n{answer}\n\n"
                    f"Refined Answer:\n\n{refined_answer}\n\n"
                    f"Execution time: {execution_time:.2f} seconds"
                )
            else:
                return "No documents retrieved for the query."
        else:
            return "Failed to build vector database."

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


# Gradio Interface
def gradio_interface(query: str) -> str:
    """
    Gradio interface function for processing queries.
    
    Args:
        query (str): The query provided by the user.
    
    Returns:
        str: ProcessReturns:
        str: Processed response to the user query.
    """
    return query_processor(query, CONFIG)


iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="Advanced Document Q&A System",
    description="This system uses a vector database to retrieve relevant information and generate answers to your questions.",
    theme="huggingface",
    css="""
    .gradio-container { 
        background-color: #f0f0f0;
        font-family: 'Arial', sans-serif;
    }
    .input-box { 
        border: 2px solid #007bff;
        border-radius: 5px;
    }
    .output-box {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
    }
    """,
    article="""
    ## How to use
    1. Enter your question in the input box.
    2. Click 'Submit' or press Enter.
    3. Wait for the system to process your query and generate an answer.
    4. The initial and refined answers will appear in the output box below, along with the execution time.

    ## About
    This system uses advanced natural language processing techniques to understand your query, 
    retrieve relevant information from a document database, and generate comprehensive answers.
    It also includes a refinement step to improve the quality of the response.
    """
)

if __name__ == "__main__":
    iface.launch(share=True, inbrowser=True, server_port=1440)