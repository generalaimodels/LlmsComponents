```python
import os
import pandas as pd
from typing import Optional, List, Dict, Any
from datasets import load_dataset
from tqdm.notebook import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Constants
MARKDOWN_SEPARATORS = ["\n\n", "\n", " "]
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
DATASET_PATH = "path/of/dataset"

# Load dataset
dataset = load_dataset(DATASET_PATH)

# Convert the dataset to LangchainDocument format
RAW_KNOWLEDGE_BASE: List[LangchainDocument] = [
    LangchainDocument(page_content=doc['text'], metadata=dict(doc))
    for doc in tqdm(dataset)
]

# Tokenizer for text splitting
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer: Optional[str] = EMBEDDING_MODEL_NAME
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed.extend(text_splitter.split_documents([doc]))

    # Remove duplicates
    unique_texts: Dict[str, bool] = {}
    docs_processed_unique: List[LangchainDocument] = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# Split and process the documents
docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# Create FAISS knowledge vector database
KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

# Function to query the vector database
def query_knowledge_base(user_query: str) -> List[LangchainDocument]:
    """
    Query the knowledge vector database and return the top matching documents.
    """
    query_vector = embedding_model.embed_query(user_query)
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=1)
    return retrieved_docs

# Load the reader model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
reader_tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=reader_tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500
)

def generate_answer(prompt: str) -> str:
    """
    Generate an answer using the reader model.
    """
    return READER_LLM(prompt)[0]["generated_text"]

# Example usage
user_query = "How to create a pipeline object?"
retrieved_docs = query_knowledge_base(user_query)
retrieved_docs_text = [doc.page_content for doc in retrieved_docs]

context = "\nExtracted documents:\n" + "".join(
    [f"Document {i}::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
)

RAG_PROMPT_TEMPLATE = reader_tokenizer.apply_chat_template(
    List[str], tokenize=False, add_generation_prompt=True
)

final_prompt = RAG_PROMPT_TEMPLATE.format(
    question=user_query, context=context
)

answer = generate_answer(final_prompt)
print(answer)

```
```python
import os
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MARKDOWN_SEPARATORS = ["\n\n", "\n", " ", ""]


def load_documents(folder_path: str) -> List[Document]:
    """
    Load documents from various file formats in the specified folder.
    """
    documents = []
    supported_extensions = {'.txt', '.csv', '.json', '.jsonl', '.parquet'}
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        _, extension = os.path.splitext(filename)
        
        if extension not in supported_extensions:
            continue
        
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={'source': filename}))
        elif extension in {'.csv', '.json', '.jsonl', '.parquet'}:
            df = pd.read_csv(file_path) if extension == '.csv' else pd.read_json(file_path, lines=(extension == '.jsonl'))
            for _, row in df.iterrows():
                documents.append(Document(page_content=str(row.to_dict()), metadata={'source': filename}))
    
    return documents


def split_documents(
    chunk_size: int,
    knowledge_base: List[Document],
    tokenizer: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[Document]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = text_splitter.split_documents(knowledge_base)

    # Remove duplicates
    unique_texts = set()
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)

    return docs_processed_unique


def create_vector_database(docs_processed: List[Document]) -> FAISS:
    """
    Create a vector database from processed documents.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )


def setup_reader_model():
    """
    Set up the reader model for text generation.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL_NAME, quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    return pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )


def generate_answer(
    user_query: str,
    knowledge_vector_database: FAISS,
    reader_llm: pipeline,
    tokenizer: AutoTokenizer,
    k: int = 1
) -> str:
    """
    Generate an answer based on the user query and retrieved documents.
    """
    retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=k)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n" + "".join(
        [f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)]
    )

    prompt_in_chat_format = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {user_query}\n\nContext: {context}"},
    ]
    rag_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    answer = reader_llm(rag_prompt_template)[0]["generated_text"]
    return answer


def main():
    # Load and process documents
    documents = load_documents("path/to/document/folder")
    processed_docs = split_documents(chunk_size=512, knowledge_base=documents)

    # Create vector database
    knowledge_vector_database = create_vector_database(processed_docs)

    # Set up reader model
    reader_llm = setup_reader_model()
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    # Example usage
    user_query = "How to create a pipeline object?"
    answer = generate_answer(user_query, knowledge_vector_database, reader_llm, tokenizer)
    print(answer)


if __name__ == "__main__":
    main()

```

```python
# Import necessary modules
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


def load_knowledge_base(data_path: str) -> List[Document]:
    """
    Load and return a list of Langchain Document objects from the dataset.
    """
    dataset = load_dataset(data_path)
    raw_knowledge_base = [
        Document(page_content=doc['text'], metadata=doc['metadata'])
        for doc in tqdm(dataset)
    ]
    return raw_knowledge_base


def split_documents(
    chunk_size: int,
    knowledge_base: List[Document],
    tokenizer_name: Optional[str] = "thenlper/gte-small",
    separators: Optional[List[str]] = ["\n\n", "\n", " "]
) -> List[Document]:
    """
    Split documents into chunks of a maximum size `chunk_size` tokens and return a list of unique documents.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=separators,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def create_knowledge_vector_database(docs_processed: List[Document], embedding_model_name: str) -> FAISS:
    """
    Create and return a FAISS vector database from processed documents.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    knowledge_vector_database = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return knowledge_vector_database


def get_retrieved_docs(
    knowledge_vector_database: FAISS, user_query: str, embedding_model_name: str
) -> List[Document]:
    """
    Retrieve and return the most relevant documents based on the user query.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    query_vector = embedding_model.embed_query(user_query)
    retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=1)
    return retrieved_docs


def generate_answer(retrieved_docs: List[Document], user_query: str, reader_model_name: str) -> str:
    """
    Generate and return an answer based on retrieved documents and user query using a language model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        reader_model_name, quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(reader_model_name)

    reader_llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {i}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    prompt_in_chat_format = [
        f"Question: {user_query}",
        f"Context: {context}"
    ]
    final_prompt = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    answer = reader_llm(final_prompt)[0]["generated_text"]
    return answer


if __name__ == "__main__":
    # Load knowledge base
    data_path = "path/of/dataset"
    knowledge_base = load_knowledge_base(data_path)

    # Split documents
    chunk_size = 512
    docs_processed = split_documents(chunk_size, knowledge_base)

    # Create knowledge vector database
    embedding_model_name = "thenlper/gte-small"
    knowledge_vector_database = create_knowledge_vector_database(docs_processed, embedding_model_name)

    # Retrieve documents
    user_query = "How to create a pipeline object?"
    retrieved_docs = get_retrieved_docs(knowledge_vector_database, user_query, embedding_model_name)

    # Generate an answer
    reader_model_name = "HuggingFaceH4/zephyr-7b-beta"
    answer = generate_answer(retrieved_docs, user_query, reader_model_name)
    print(answer)

```