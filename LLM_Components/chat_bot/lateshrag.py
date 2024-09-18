import sys
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import faiss
import gradio as gr
from transformers import AutoTokenizer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from g4f.client import Client

from datacollection import AdvancedDirectoryLoader
from document_splitter import AdvancedDocumentSplitter
from embedding_data import AdvancedFAISS

# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CONFIG = {
    "EMBEDDING_MODEL_NAME": "thenlper/gte-large",
    "DATA_DIR": Path(r"C:\Users\heman\Desktop\Coding\LlmsComponents\LLM_Components\chat_bot"),
    "CHUNK_SIZE": 512,
    "K": 5,  # Number of similar documents to retrieve
    "LLM_MODEL": "gpt-4o-mini",
}

client = Client()


def process_document(doc: str, splitter: AdvancedDocumentSplitter) -> List[Dict[str, Any]]:
    """Process a single document using the document splitter."""
    return splitter.split_documents([doc])


def build_vector_database(
    data_dir: Path,
    embedding_model_name: str,
    chunk_size: int,
) -> Optional[AdvancedFAISS]:
    """Build a vector database from documents in the data directory."""
    try:
        loader = AdvancedDirectoryLoader(
            str(data_dir), exclude=[".pyc", "__pycache__"], silent_errors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        splitter = AdvancedDocumentSplitter(tokenizer=tokenizer, chunk_size=chunk_size)

        documents = loader.load()

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

        return advanced_faiss.from_documents(
            docs_processed, embeddings_model, distance_strategy=DistanceStrategy.COSINE
        )

    except Exception as e:
        print(f"Error building vector database: {e}")
        return None


def format_prompt(retrieved_docs: List[Dict[str, Any]], question: str) -> str:
    """Format the prompt based on retrieved documents and the user question."""
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n" + "".join(
        [f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)]
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


class VectorDatabase:
    """Singleton class to manage the vector database."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDatabase, cls).__new__(cls)
            cls._instance.db = build_vector_database(
                CONFIG["DATA_DIR"], CONFIG["EMBEDDING_MODEL_NAME"], CONFIG["CHUNK_SIZE"]
            )
        return cls._instance

    def get_db(self):
        return self.db


def query_processor(user_query: str) -> str:
    """Process the user query and return the response."""
    try:
        start_time = time.time()

        vector_db = VectorDatabase().get_db()

        if vector_db:
            retrieved_docs = vector_db.similarity_search(user_query, k=CONFIG["K"])

            if retrieved_docs:
                final_prompt = format_prompt(retrieved_docs, user_query)

                response = client.chat.completions.create(
                    model=CONFIG["LLM_MODEL"],
                    messages=[{"role": "user", "content": final_prompt}],
                )
                answer = response.choices[0].message.content

                rethinker = f"Answer to this best of your ability: {answer}"
                response = client.chat.completions.create(
                    model=CONFIG["LLM_MODEL"],
                    messages=[{"role": "user", "content": rethinker}],
                )
                refined_answer = response.choices[0].message.content

                execution_time = time.time() - start_time

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


def gradio_interface(query: str) -> str:
    """Gradio interface function for processing queries."""
    return query_processor(query)


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
    1. Enter your question in the query input box.
    2. Adjust the configuration parameters if needed:
       - Embedding Model Name: The name of the model used for embedding documents.
       - Data Directory: The path to the directory containing your documents.
       - Chunk Size: The size of text chunks for processing.
       - K: The number of similar documents to retrieve.
       - LLM Model: The language model used for generating answers.
    3. Click 'Submit' or press Enter.
    4. Wait for the system to process your query and generate an answer.
    5. The initial and refined answers will appear in the output box below, along with the execution time.

    ## About
    This system uses advanced natural language processing techniques to understand your query, 
    retrieve relevant information from a document database, and generate comprehensive answers.
    It also includes a refinement step to improve the quality of the response.
    """
)

if __name__ == "__main__":
    iface.launch(share=True, inbrowser=True, server_port=1440)