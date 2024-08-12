from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from langchain_huggingface import HuggingFaceEmbeddings
from AgentPipeline.agentdatacollection import (
    ConvertFolder_to_TxtFolder,
    AgentDataset,
    AgentDatasetLoader,
    AgentRAGPipeline
)
from AgentPipeline.agentdataretrieval import AgentContentRetrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAgentAutomationPipeline:
    def __init__(
        self,
        input_folders: Union[str, List[str]],
        output_folder: str,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        model_kwargs: Dict[str, Any] = {'device': 'cpu'},
        encode_kwargs: Dict[str, Any] = {'normalize_embeddings': False},
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_workers: int = 4
    ):
        self.input_folders = [Path(folder) for folder in (input_folders if isinstance(input_folders, list) else [input_folders])]
        self.output_folder = Path(output_folder)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self.agent_dataset = None
        self.agent_loader = None
        self.agent_embedding = None
        self.agent_rag = None

    def prepare_data(self) -> None:
        """Convert input folders to txt folders and load the dataset."""
        logger.info("Preparing data...")
        self.output_folder = ConvertFolder_to_TxtFolder(
            input_folders=[str(folder) for folder in self.input_folders],
            output_folder=str(self.output_folder)
        )
        self.agent_dataset = AgentDataset(source=str(self.output_folder)).load_data()

    def load_data(self) -> None:
        """Load all datasets."""
        if not self.agent_dataset:
            raise ValueError("Dataset not prepared. Call prepare_data() first.")

        logger.info("Loading data...")
        self.agent_loader = AgentDatasetLoader(dataset=self.agent_dataset)
        files = list(map(str, self.agent_dataset.keys()))
        self.all_page_contents, self.all_metadatas = self.agent_loader.load_multiple_datasets(files)

    def create_embedding(self) -> None:
        """Create embedding for the loaded data."""
        if not self.agent_loader:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Creating embeddings...")
        self.agent_embedding = AgentContentRetrieval(embedding_model=self.embedding_model)
        self.agent_embedding.add_documents(self.all_page_contents, self.all_metadatas)

    def setup_rag_pipeline(self) -> None:
        """Set up the RAG pipeline."""
        logger.info("Setting up RAG pipeline...")
        self.agent_rag = AgentRAGPipeline(
            embedding_model=self.embedding_model,
            directory_path=str(self.output_folder),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.agent_rag.load_documents()
        self.agent_rag.split_documents()
        self.agent_rag.create_vectorstore()

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query."""
        if not self.agent_embedding:
            raise ValueError("Embedding not created. Call create_embedding() first.")

        logger.info(f"Retrieving documents for query: {query}")
        results = self.agent_embedding.retrieve(query=query, top_k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in results]

    def rag_query(self, query: str) -> List[str]:
        """Query using the RAG pipeline."""
        if not self.agent_rag:
            raise ValueError("RAG pipeline not set up. Call setup_rag_pipeline() first.")

        logger.info(f"Querying RAG pipeline: {query}")
        return self.agent_rag.query(query=query)

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset."""
        if not self.agent_loader:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.agent_loader.get_dataset_summary()

    def process_query_batch(self, queries: List[str], method: str = 'retrieve', top_k: int = 2) -> List[Any]:
        """Process a batch of queries using either retrieve or rag_query method."""
        if method not in ['retrieve', 'rag_query']:
            raise ValueError("Invalid method. Choose either 'retrieve' or 'rag_query'.")

        logger.info(f"Processing batch of {len(queries)} queries using {method} method...")
        func = partial(self.retrieve, top_k=top_k) if method == 'retrieve' else self.rag_query

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {executor.submit(func, query): query for query in queries}
            results = []
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    results.append({"query": query, "result": result})
                except Exception as exc:
                    logger.error(f"Query {query} generated an exception: {exc}")
                    results.append({"query": query, "error": str(exc)})
        return results

    def run_pipeline(self, queries: List[str], method: str = 'retrieve', top_k: int = 2) -> List[Any]:
        """Run the complete pipeline."""
        logger.info("Starting pipeline...")
        self.prepare_data()
        self.load_data()
        self.create_embedding()
        if method == 'rag_query':
            self.setup_rag_pipeline()
        return self.process_query_batch(queries, method, top_k)

def main():
    # Example usage
    pipeline = AdvancedAgentAutomationPipeline(
        input_folders=[r"C:\Users\heman\Desktop\components\LlmsComponents\LLM_Components\AgentPipeline\agentcallingpipline"],
        output_folder="./output",
        max_workers=4
    )

    queries = [
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline"
    ]

    # Run pipeline with retrieve method
    retrieve_results = pipeline.run_pipeline(queries, method='retrieve', top_k=1)
    for result in retrieve_results:
        print(f"Query: {result['query']}")
        print(f"Results: {result['result']}")
        print("---")

    # Run pipeline with rag_query method
    rag_results = pipeline.run_pipeline(queries, method='rag_query')
    for result in rag_results:
        print(f"Query: {result['query']}")
        print(result['result'])


if __name__ == "__main__":
    main()