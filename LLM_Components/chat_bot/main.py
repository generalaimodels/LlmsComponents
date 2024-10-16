import os
import sys
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
import faiss
from pathlib import Path
from langchain_community.docstore.in_memory import InMemoryDocstore
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from datacollection import AdvancedDirectoryLoader
from document_splitter import AdvancedDocumentSplitter
from embedding_data import AdvancedFAISS
from g4f.client import Client

# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configuration - Consider using environment variables or a configuration file
CONFIG = {
    "EMBEDDING_MODEL_NAME": "thenlper/gte-large",
    "DATA_DIR": r"E:\LLMS\Fine-tuning\LlmsComponents\LLM_Components\chat_bot",
    "CHUNK_SIZE": 512,
    "K": 5,  # Number of similar documents to retrieve
    "LLM_MODEL": "gpt-4o-mini",
}

client = Client()


def process_document(doc: str, splitter: AdvancedDocumentSplitter) -> List[Dict[str, Any]]:
    """
    Process a single document using the document splitter.
    """
    return splitter.split_documents([doc])


def build_vector_database(
    data_dir: str, embedding_model_name: str, chunk_size: int, k: int
) -> Optional[List[Dict[str, Any]]]:
    """
    Build a vector database from documents in the data directory.
    """
    try:
        loader = AdvancedDirectoryLoader(data_dir, exclude=[".pyc", "__pycache__"], silent_errors=True)
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        splitter = AdvancedDocumentSplitter(tokenizer=tokenizer, chunk_size=chunk_size)

        documents = loader.load()

        # Process documents sequentially
        docs_processed = [process_document(doc, splitter) for doc in documents]
        docs_processed = [doc for sublist in docs_processed for doc in sublist]

        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
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
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
        )

        faiss_index = advanced_faiss.from_documents(
            docs_processed, embeddings_model, distance_strategy=DistanceStrategy.COSINE
        )

        return faiss_index

    except Exception as e:
        print(f"Error building vector database: {e}")
        return None


def format_prompt(retrieved_docs: List[Dict[str, Any]], question: str) -> str:
    """
    Format the prompt based on retrieved documents and the user question.
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
If the answer cannot be deduced from the context, Always try to give best of ability"""
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


def main(user_query: str, config: Dict[str, Any]) -> None:
    """
    Main function to orchestrate the process.
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


if __name__ == "__main__":
    user_query = """
    Write a document loading pipeline in Python. Write in great detail.
    """
    main(user_query, CONFIG)