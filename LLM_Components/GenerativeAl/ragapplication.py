import os
from typing import List, Dict, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers.agents import Tool, HfEngine, ReactJsonAgent
from tqdm import tqdm


def load_knowledge_base(dataset_name: str, dataset_path: Optional[str] = None):
    """
    Loads the specified dataset from HuggingFace datasets library.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        dataset_path (Optional[str]): Path of the folder for dataset (if any).
    
    Returns:
        knowledge_base: Loaded dataset object.
    """
    if dataset_path:
        knowledge_base = load_dataset(dataset_name, data_files={'train': dataset_path})
    else:
        knowledge_base = load_dataset(dataset_name)
    
    return knowledge_base


def create_documents(knowledge_base) -> List[Document]:
    """
    Converts the dataset into a list of Document objects.
    
    Args:
        knowledge_base: The dataset loaded using `load_dataset`.
    
    Returns:
        List[Document]: List of Document objects.
    """
    return [
        Document(page_content=doc['text'], metadata={'source': doc['source']})
        for doc in knowledge_base['train']
    ]


def split_and_filter_documents(source_docs: List[Document], tokenizer_name: str, chunk_size: int = 200, chunk_overlap: int = 20) -> List[Document]:
    """
    Splits documents into smaller chunks and filters out duplicates.
    
    Args:
        source_docs (List[Document]): List of source Document objects.
        tokenizer_name (str): Name of the tokenizer to use.
        chunk_size (int, optional): The size of each chunk.
        chunk_overlap (int, optional): The overlap between chunks.
    
    Returns:
        List[Document]: List of processed Document objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    unique_texts = {}

    print("Splitting documents...")
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)
    
    return docs_processed


def create_vectordb(docs_processed: List[Document], model_name: str) -> FAISS:
    """
    Creates a FAISS vector database from processed documents.
    
    Args:
        docs_processed (List[Document]): List of processed Document objects.
        model_name (str): Name of the embedding model.
    
    Returns:
        FAISS: The FAISS vector database.
    """
    print("Embedding documents... This may take a few minutes.")
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return vectordb


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves documents from the knowledge base with the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, vectordb: FAISS, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "The search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {i} =====\n" + doc.page_content + "\n"
                for i, doc in enumerate(docs)
            ]
        )


def main(query: str):
    dataset_name = "your_dataset_name"
    dataset_path = "path/of/folder"
    
    # Load knowledge base
    knowledge_base = load_knowledge_base(dataset_name, dataset_path)
    
    # Create source documents
    source_docs = create_documents(knowledge_base)
    
    # Split and filter documents
    docs_processed = split_and_filter_documents(
        source_docs=source_docs,
        tokenizer_name="thenlper/gte-small"
    )
    
    # Create FAISS vector database
    vectordb = create_vectordb(
        docs_processed=docs_processed,
        model_name="thenlper/gte-small"
    )
    
    # Initialize and run the agent
    retriever_tool = RetrieverTool(vectordb)
    llm_engine = HfEngine("CohereForAI/c4ai-command-r-plus")
    agent = ReactJsonAgent(
        tools=[retriever_tool], llm_engine=llm_engine, max_iterations=4, verbose=2
    )
    
    agent_output = agent.run(query)
    
    print("Final output:")
    print(agent_output)


if __name__ == "__main__":
    query = "Your search query here"
    main(query)
    
    


import typing
from datasets import load_dataset
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import Tool, HfEngine, ReactJsonAgent
from langchain_core.vectorstores import VectorStore

def load_knowledge_base(dataset_name: str, path: str) -> typing.Dict:
    """
    Load the knowledge base dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        path (str): Path to the dataset folder.
    
    Returns:
        Dict: Loaded dataset.
    """
    return load_dataset(dataset_name, path)

def process_documents(knowledge_base: typing.Dict) -> typing.List[Document]:
    """
    Process the knowledge base into Document objects.
    
    Args:
        knowledge_base (Dict): The loaded knowledge base.
    
    Returns:
        List[Document]: List of processed Document objects.
    """
    return [
        Document(page_content=str(doc['content']), metadata={'source': doc['source']})
        for doc in knowledge_base
    ]

def split_documents(source_docs: typing.List[Document]) -> typing.List[Document]:
    """
    Split documents and remove duplicates.
    
    Args:
        source_docs (List[Document]): List of source documents.
    
    Returns:
        List[Document]: List of unique, split documents.
    """
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    print("Splitting documents...")
    docs_processed = []
    unique_texts = set()
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts.add(new_doc.page_content)
                docs_processed.append(new_doc)
    
    return docs_processed

def create_vector_db(docs_processed: typing.List[Document]) -> FAISS:
    """
    Create a vector database from processed documents.
    
    Args:
        docs_processed (List[Document]): List of processed documents.
    
    Returns:
        FAISS: Vector database.
    """
    print("Embedding documents... This may take a few minutes.")
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    return FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

class RetrieverTool(Tool):
    """Custom retriever tool for the agent."""

    name = "retriever"
    description = "Retrieves documents from the knowledge base using semantic similarity."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. Use affirmative form for best results.",
        }
    }
    output_type = "text"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("Search query must be a string.")

        docs = self.vectordb.similarity_search(query, k=7)
        return "\nRetrieved documents:\n" + "\n".join(
            f"===== Document {i} =====\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

def setup_agent(vectordb: FAISS) -> ReactJsonAgent:
    """
    Set up the agent with the retriever tool.
    
    Args:
        vectordb (FAISS): Vector database for retrieval.
    
    Returns:
        ReactJsonAgent: Configured agent.
    """
    llm_engine = HfEngine("CohereForAI/c4ai-command-r-plus")
    retriever_tool = RetrieverTool(vectordb)
    return ReactJsonAgent(
        tools=[retriever_tool], llm_engine=llm_engine, max_iterations=4, verbose=2
    )

def main():
    # Load and process the knowledge base
    knowledge_base = load_knowledge_base("your_dataset_name", "path/to/your/dataset")
    source_docs = process_documents(knowledge_base)
    docs_processed = split_documents(source_docs)
    vectordb = create_vector_db(docs_processed)

    # Set up and run the agent
    agent = setup_agent(vectordb)
    query = input("Enter your query: ")
    agent_output = agent.run(query)

    print("Final output:")
    print(agent_output)

if __name__ == "__main__":
    main()