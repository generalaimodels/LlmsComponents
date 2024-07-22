from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
import datasets
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from transformers import PreTrainedTokenizerBase
import hashlib

from typing import List, Any
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import sys
from typing import List, Any
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    Pipeline,
)


def create_reader_llm(
    reader_model_name: str,
    cache_dir: Optional[str] = None,
    temperature: float = 0.2,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 500,
) -> Pipeline:
    """
    Create a reader LLM pipeline using the specified model.

    Args:
        reader_model_name (str): The name of the reader model to use.
        cache_dir (Optional[str]): Directory to cache the model files.
        temperature (float): Sampling temperature for text generation.
        repetition_penalty (float): Penalty for token repetition.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        Pipeline: The configured text generation pipeline.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        reader_model_name,
        quantization_config=bnb_config,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        reader_model_name,
        cache_dir=cache_dir,
    )

    reader_llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )

    return reader_llm



def initialize_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Initializes the HuggingFace embedding model with the specified model name.

    Args:
        model_name (str): The name of the embedding model to use.

    Returns:
        HuggingFaceEmbeddings: An instance of the HuggingFaceEmbeddings class.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}  # Set `True` for cosine similarity
        )
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        sys.exit(1)

def create_knowledge_vector_database(docs_processed: List[str], embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    Creates a knowledge vector database from processed documents.

    Args:
        docs_processed (List[str]): The documents to be processed.
        embedding_model (HuggingFaceEmbeddings): The embedding model to use.

    Returns:
        FAISS: An instance of the FAISS vector store.
    """
    try:
        return FAISS.from_documents(
            docs_processed, 
            embedding_model, 
            distance_strategy=DistanceStrategy.COSINE
        )
    except Exception as e:
        print(f"Error creating knowledge vector database: {e}")
        sys.exit(1)

def retrieve_documents(knowledge_vector_database: FAISS, user_query: str, k: int) -> List[Any]:
    """
    Retrieves the top k documents based on the user's query.

    Args:
        knowledge_vector_database (FAISS): The knowledge vector database.
        user_query (str): The query string.
        k (int): The number of top documents to retrieve.

    Returns:
        List[Any]: A list of retrieved documents.
    """
    try:
        query_vector = knowledge_vector_database.embed_query(user_query)
        print(f"\nStarting retrieval for user_query='{user_query}'...")
        retrieved_docs = knowledge_vector_database.similarity_search(query=query_vector, k=k)
        return retrieved_docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        sys.exit(1)

def main(embedding_model_name: str, docs_processed: List[str], k: int) -> None:
    """
    Main function to initialize the embedding model, create the knowledge vector database,
    and retrieve documents based on the user query.

    Args:
        embedding_model_name (str): The name of the embedding model.
        docs_processed (List[str]): The processed documents.
        k (int): The number of top documents to retrieve.
    """
    embedding_model = initialize_embedding_model(embedding_model_name)
    knowledge_vector_database = create_knowledge_vector_database(docs_processed, embedding_model)
    
    # Example user query; this could be replaced with user input
    user_query = "What is your query here?"  # Placeholder query
    retrieved_docs = retrieve_documents(knowledge_vector_database, user_query, k)
    
    print("\n==================================Top document==================================")
    print(retrieved_docs[0].page_content)
    print("==================================Metadata==================================")
    print(retrieved_docs[0].metadata)


# Constants
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
MARKDOWN_SEPARATORS = ["\n", " ", ".", ","]  # Define your separators as needed

class DocumentChunker:
    """
    A class to split documents into chunks based on a specified token size.
    """

    def __init__(self, chunk_size: int, tokenizer: AutoTokenizer):
        """
        Initialize the DocumentChunker with a specified chunk size and tokenizer.

        Args:
            chunk_size (int): The maximum number of tokens per document chunk.
            tokenizer (AutoTokenizer): The tokenizer used for splitting the documents.
        """
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

    def split_documents(self, knowledge_base: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        Split documents into chunks and return a list of unique processed documents.

        Args:
            knowledge_base (List[LangchainDocument]): The list of documents to be processed.

        Returns:
            List[LangchainDocument]: A list of unique processed document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size // 10,  # 10% overlap
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed.extend(text_splitter.split_documents([doc]))

        # Remove duplicates based on page_content
        unique_texts = set()
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

        return docs_processed_unique

MARKDOWN_SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    ""
]

class DocumentProcessor:
    def __init__(self, embedding_model_name: str):
        self.embedding_model_name = embedding_model_name
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load the tokenizer for the specified embedding model."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.embedding_model_name)

    def split_documents(
        self,
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of unique documents.

        Args:
            chunk_size (int): Maximum number of tokens per chunk.
            knowledge_base (List[LangchainDocument]): List of input documents.

        Returns:
            List[LangchainDocument]: List of processed and unique documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed.extend(text_splitter.split_documents([doc]))

        return self._remove_duplicates(docs_processed)

    @staticmethod
    def _remove_duplicates(docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """Remove duplicate documents based on their content hash."""
        unique_docs = {}
        for doc in docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in unique_docs:
                unique_docs[content_hash] = doc
        return list(unique_docs.values())


def load_and_process_dataset(dataset_path: str) -> List[LangchainDocument]:
    """
    Load and process a dataset, combining all columns into page_content
    and creating metadata with column names and corresponding text.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        List[LangchainDocument]: List of processed documents.
    """
    try:
        # Load the dataset
        ds = load_dataset(dataset_path)
        if 'train' not in ds:
            raise KeyError("Dataset does not contain a 'train' split.")
        
        ds = ds['train']
        columns = ds.column_names

        # Process the dataset
        raw_knowledge_base = []
        for doc in tqdm(ds, desc="Processing documents"):
            page_content = " ".join(str(doc[col]) for col in columns)
            metadata = {col: str(doc[col]) for col in columns}
            
            raw_knowledge_base.append(
                LangchainDocument(page_content=page_content, metadata=metadata)
            )

        return raw_knowledge_base

    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")
        return []
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer, Pipeline
from langchain.schema import Document

def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    reader_llm: Pipeline,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = "You are an advanced AI agent capable of answering questions based on provided context.",
    max_context_length: int = 2048
) -> str:
    """
    Generate an answer based on the query and retrieved documents using a language model.

    Args:
        query (str): The user's question.
        retrieved_docs (List[Document]): List of retrieved documents.
        reader_llm (Pipeline): The language model pipeline for text generation.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.
        system_prompt (str): The system prompt to guide the model's behavior.
        max_context_length (int): Maximum number of tokens for the context.

    Returns:
        str: The generated answer.
    """
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": "Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}",
        },
    ]

    rag_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    context = _prepare_context(retrieved_docs, max_context_length, tokenizer)
    
    final_prompt = rag_prompt_template.format(question=query, context=context)

    answer = reader_llm(final_prompt)[0]["generated_text"]
    return answer
def _prepare_context(
    retrieved_docs: List[Document],
    max_context_length: int,
    tokenizer: PreTrainedTokenizer
) -> str:
    """
    Prepare the context from retrieved documents, ensuring it doesn't exceed the maximum length.

    Args:
        retrieved_docs (List[Document]): List of retrieved documents.
        max_context_length (int): Maximum number of tokens for the context.
        tokenizer (PreTrainedTokenizer): The tokenizer for the language model.

    Returns:
        str: The prepared context string.
    """
    context = "\nExtracted documents:\n"
    current_length = len(tokenizer.encode(context))

    for i, doc in enumerate(retrieved_docs):
        doc_text = f"Document {i}:::\n{doc.page_content}\n"
        doc_length = len(tokenizer.encode(doc_text))
        
        if current_length + doc_length > max_context_length:
            break
        
        context += doc_text
        current_length += doc_length

    return context
class DatasetProcessor:
    """
    A class to process datasets and return a RAW_KNOWLEDGE_BASE.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the DatasetProcessor with the dataset path.

        Args:
            dataset_path (str): The path of the dataset to load.
        """
        self.dataset_path = dataset_path
        self.ds = self.load_dataset()

    def load_dataset(self) -> Any:
        """
        Load the dataset from the specified path.

        Returns:
            Any: The loaded dataset.
        """
        try:
            return datasets.load_dataset(self.dataset_path)['train']
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {self.dataset_path}: {e}")

    def combine_columns(self) -> List[LangchainDocument]:
        """
        Combine columns of the dataset into page content and create metadata.

        Returns:
            List[LangchainDocument]: A list of documents with combined content and metadata.
        """
        columns = self.ds.column_names
        raw_knowledge_base = []

        for doc in tqdm(self.ds, desc="Processing documents", unit="doc"):
            page_content = " ".join(str(doc.get(col, '')) for col in columns)
            metadata = {col: doc.get(col, '') for col in columns}  # Create metadata dictionary
            
            raw_knowledge_base.append(
                LangchainDocument(page_content=page_content, metadata=metadata)
            )

        return raw_knowledge_base


def main(dataset_path: str) -> List[LangchainDocument]:
    """
    Main function to create a DatasetProcessor instance and retrieve the RAW_KNOWLEDGE_BASE.

    Args:
        dataset_path (str): The path of the dataset to process.

    Returns:
        List[LangchainDocument]: The resulting RAW_KNOWLEDGE_BASE.
    """
    processor = DatasetProcessor(dataset_path)
    return processor.combine_columns()

