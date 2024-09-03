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
from g4f.client import Client

from .datacollection import AdvancedDirectoryLoader
from .document_splitter import AdvancedDocumentSplitter
from .embedding_data import AdvancedFAISS



# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

client = Client()

def process_document(doc: str, splitter: AdvancedDocumentSplitter) -> List[Dict[str, Any]]:
    """Process a single document using the document splitter."""
    return splitter.split_documents([doc])

def build_vector_database(data_dir: str, embedding_model_name: str, chunk_size: int, k: int) -> Optional[AdvancedFAISS]:
    """Build a vector database from documents in the data directory."""
    try:
        loader = AdvancedDirectoryLoader(data_dir, exclude=[".pyc", "__pycache__"], silent_errors=True)
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        splitter = AdvancedDocumentSplitter(tokenizer=tokenizer, chunk_size=chunk_size)

        documents = loader.load()
        docs_processed = [doc for sublist in [process_document(doc, splitter) for doc in documents] for doc in sublist]

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

def format_prompt(retrieved_docs: List[Dict[str, Any]], question: str) -> str:
    """Format the prompt based on retrieved documents and the user question."""
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


# # Build the vector database once
# vector_db = build_vector_database(config)

# def query_processor(user_query: str) -> str:
#     """Process the user query and return the response."""
#     try:
#         start_time = time.time()

#         if vector_db:
#             # Perform similarity search
#             retrieved_docs = vector_db.similarity_search(user_query, k=config.K)

#             if retrieved_docs:
#                 # Format the prompt
#                 final_prompt = format_prompt(retrieved_docs, user_query)

#                 # Get LLM response
#                 response = client.chat.completions.create(
#                     model=config.LLM_MODEL,
#                     messages=[{"role": "user", "content": final_prompt}],
#                 )
#                 answer = response.choices[0].message.content

#                 # Rethinker - Optional step for refining the answer
#                 rethinker = f"Answer to this best of your ability: {answer}"
#                 response = client.chat.completions.create(
#                     model=config.LLM_MODEL,
#                     messages=[{"role": "user", "content": rethinker}],
#                 )
#                 refined_answer = response.choices[0].message.content

#                 end_time = time.time()
#                 execution_time = end_time - start_time

#                 return (
#                     f"Initial Answer:\n\n{answer}\n\n"
#                     f"Refined Answer:\n\n{refined_answer}\n\n"
#                     f"Execution time: {execution_time:.2f} seconds"
#                 )
#             else:
#                 return "No documents retrieved for the query."
#         else:
#             return "Failed to build vector database."

#     except Exception as e:
#         return f"An unexpected error occurred: {str(e)}"