from typing import List, Optional, Tuple, Dict
from pathlib import Path
import json
import csv
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import torch


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MARKDOWN_SEPARATORS = ["\n\n", "\n", " ", ""]


def load_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Load documents from various file formats in the specified folder.
    """
    documents = []
    for file_path in Path(folder_path).glob("*"):
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({"content": f.read(), "source": str(file_path)})
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                documents.append({"content": " ".join(row.astype(str)), "source": str(file_path)})
        elif file_path.suffix in (".json", ".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    documents.append({"content": json.dumps(data), "source": str(file_path)})
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
            for _, row in df.iterrows():
                documents.append({"content": " ".join(row.astype(str)), "source": str(file_path)})
    return documents


def create_langchain_documents(documents: List[Dict[str, str]]) -> List[Document]:
    """
    Create Langchain Document objects from the loaded documents.
    """
    return [
        Document(page_content=doc["content"], metadata={"source": doc["source"]})
        for doc in tqdm(documents,desc="Creating Documents:")
    ]


def split_documents(
    chunk_size: int,
    knowledge_base: List[Document],
    tokenizer: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[Document]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer),
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
    for doc in tqdm(docs_processed,desc="Chunking Documents:"):
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)

    return docs_processed_unique


def create_vector_database(docs_processed: List[Document]) -> FAISS:
    """
    Create a FAISS vector database from processed documents.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        force_download=False,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )


def setup_reader_model( ):
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
        pretrained_model_name_or_path=READER_MODEL_NAME, quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained( pretrained_model_name_or_path=READER_MODEL_NAME)

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
    query: str,
    knowledge_base: FAISS,
    reader_llm: pipeline,
    tokenizer: AutoTokenizer,
    k: int = 1,
) -> str:
    """
    Generate an answer based on the query and retrieved documents.
    """
    retrieved_docs = knowledge_base.similarity_search(query=query, k=k)
    retrieved_docs_text = [doc.page_content for doc in tqdm(retrieved_docs,desc="Retrieving Documents:")]
    context = "\nExtracted documents:\n" + "".join(
        [f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)]
    )

    prompt_in_chat_format = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {query}\n\nContext: {context}"},
    ]
    rag_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    return reader_llm(rag_prompt_template)[0]["generated_text"]


def main():
    # Load and process documents
    documents = load_documents(r"C:\Users\heman\Desktop\components\output")
    langchain_docs = create_langchain_documents(documents)
    processed_docs = split_documents(chunk_size=1024, knowledge_base=langchain_docs)
    print(len(processed_docs))
    # Create vector database
    knowledge_base = create_vector_database(processed_docs)
    query="Linux  Commands"
    k=2
    retrieved_docs = knowledge_base.similarity_search(query=query, k=k)
    print(retrieved_docs)
    # Set up reader model
    reader_llm = setup_reader_model()
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    # Example usage
    user_query = "Linux  Commands"
    answer = generate_answer(user_query, knowledge_base, reader_llm, tokenizer)
    print(f"Question: {user_query}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()