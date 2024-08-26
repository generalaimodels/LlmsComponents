import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import asyncio
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from datacollection import AdvancedDirectoryLoader
from document_splitter import AdvancedDocumentSplitter
from embedding_data import AdvancedFAISS
import sys

from g4f.client import Client

client = Client()

# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def process_document(doc: str, splitter: AdvancedDocumentSplitter) -> List[Dict[str, Any]]:
    """
    Process a single document using the document splitter.
    """
    return splitter.split_documents([doc])


def build_vector_database(
    data_dir: str, embedding_model_name: str, chunk_size: int = 512, k: int = 5
) -> List[Dict[str, Any]]:
    """
    Build a vector database from documents in the data directory.
    """
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

    return faiss_index.similarity_search(USER_QUERY, k=k)


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
If the answer cannot be deduced from the context, do not give an answer."""
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


if __name__ == "__main__":
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    DATA_DIR = r"C:\Users\heman\Desktop\Coding\LlmsComponents\LLM_Components\AgentPipeline"  
    USER_QUERY = """
    write python code using above content below query
    
    Function to load a transformers model.

        Args:
            model_type (str): The type of the model (e.g., 'causal_lm', 'masked_lm').
            model_name_or_path (Union[str, Path]): The name or path of the model.
            *model_args: Additional positional arguments to pass to the model's from_pretrained method.
            **kwargs: Additional keyword arguments to pass to the model's from_pretrained method.

        Returns:
            model (Type): The loaded model.

        Raises:
            ValueError: If the specified model type is unknown.
            ModelLoadingError: If an error occurs during model loading.
    
    """

    # Build and query the vector database
    retrieved_docs = build_vector_database(DATA_DIR, EMBEDDING_MODEL_NAME)

    if retrieved_docs:
        print(retrieved_docs[0].page_content)
        print("================================== Metadata ==================================")
        print(retrieved_docs[0].metadata)

        final_prompt = format_prompt(retrieved_docs, USER_QUERY)
        print(final_prompt,"\n \n \n")
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt}],

        )
        print("Answer:\n \n ",response.choices[0].message.content)
        # print(final_prompt)
    else:
        print(f"No documents retrieved for the query.")