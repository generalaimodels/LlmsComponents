
import datasets
from typing import List, Dict, Optional
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


from raglogger import setup_logger

logger = setup_logger()

# Constants for embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Replace with your preferred model

# Define markdown separators
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",  # Include space as separator
    "",   # Empty string to finalize small chunks
]

def create_raw_knowledge_base(dataset_name: str) -> List[LangchainDocument]:
    """
    Create a raw knowledge base from a user-specified dataset.
    Generalized to load any dataset without hardcoding specific splits.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        List[LangchainDocument]: A list of Document objects with page content and metadata.
    """
    try:
        dataset = datasets.load_dataset(dataset_name)
        raw_knowledge_base: List[LangchainDocument] = []

        for split_name in dataset.keys():
            logger.info(f"Processing split: {split_name}")
            for entry in tqdm(dataset[split_name]):
                # Concatenating all field values into a single string for `page_content`
                page_content = " ".join([str(value).strip() for value in entry.values()])
                metadata = {key: [str(value)] for key, value in entry.items()}
                document = LangchainDocument(page_content=page_content, metadata=metadata)
                raw_knowledge_base.append(document)

        return raw_knowledge_base

    except Exception as e:
        logger.error(f"An error occurred while creating the knowledge base: {e}")
        return []

def chunk_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME # Default to EMBEDDING_MODEL_NAME
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum `chunk_size` tokens using tokenization.
    
    Args:
        chunk_size (int): Maximum chunk size in tokens.
        knowledge_base (List[LangchainDocument]): List of documents to chunk.
        tokenizer_name (Optional[str]): The tokenizer model to use for chunking.
    
    Returns:
        List[LangchainDocument]: A list of chunked and deduplicated documents.
    """
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,cache_dir="./tokenizerdata")
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        return []

    # Setup the text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),  # 10% overlap to prevent information leakage
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS
    )

    docs_processed = []
    
    # Split each document into chunks
    for doc in tqdm(knowledge_base, desc="Splitting documents"):
        try:
            docs_processed += text_splitter.split_documents([doc])
        except Exception as e:
            logger.error(f"Error splitting document: {e}")

    # Remove duplicate documents
    unique_texts = {}
    docs_processed_unique = []
    
    for doc in docs_processed:
        # We strip leading/trailing whitespace to avoid duplicating documents with whitespace variations
        stripped_content = doc.page_content.strip()
        if stripped_content not in unique_texts:
            unique_texts[stripped_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

def create_knowledge_vector_database(
    dataset_name: str,
    chunk_size: int = 512,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    num_workers: int = 4
) -> Optional[FAISS]:
    """
    Creates a knowledge vector database using FAISS and the HuggingFace embeddings.

    Parameters:
        dataset_name (str): The name of the dataset.
        chunk_size (int): Size of each document chunk.
        embedding_model_name (str): The HuggingFace model name for embeddings.
        distance_strategy (DistanceStrategy): The distance strategy for FAISS.
        num_workers (int): Number of workers for multiprocessing.

    Returns:
        Optional[FAISS]: A FAISS vector store or None if creation fails.
    """
    try:
        raw_knowledge_base = create_raw_knowledge_base(dataset_name)
        if not raw_knowledge_base:
            logger.warning("Failed to create raw knowledge base.")
            return None

        # Use multiprocessing for document chunking
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_chunks = executor.submit(chunk_documents, chunk_size, raw_knowledge_base)
            processed_docs = future_chunks.result()

        if not processed_docs:
            logger.warning("No documents were processed after chunking.")
            return None
        
        logger.info(f"Successfully processed {len(processed_docs)} unique document chunks.")

        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        knowledge_vector_database = FAISS.from_documents(
            processed_docs,
            embedding_model,
            distance_strategy=distance_strategy
        )

        return knowledge_vector_database, embedding_model

    except Exception as e:
        logger.error(f"Exception occurred while creating the knowledge vector database: {e}")
        return None


# if __name__ == "__main__":
#     # Example usage
#     dataset_name = "fka/awesome-chatgpt-prompts"
    
#     knowledge_vector_database,embedding_model = create_knowledge_vector_database(dataset_name)
#     if knowledge_vector_database:
#         logger.info("Knowledge Vector Database created successfully.")
#     else:
#         logger.error("Failed to create the Knowledge Vector Database.")
#     user_query="your role is act data scientist very advanced orojects details with"
#     print(f"\nStarting retrieval for {user_query}...")
#     retrieved_docs =  knowledge_vector_database.similarity_search(query=user_query, k=10)
#     print(retrieved_docs[0].page_content)
#     print(retrieved_docs[0].metadata)
#     retrieved_docs_text = [
#     doc.page_content for doc in retrieved_docs
#     ]  # We only need the text of the documents
#     context = "\nExtracted documents:\n"
#     context += "".join(
#         [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
#     )

