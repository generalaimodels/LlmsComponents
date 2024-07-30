import os
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime
from LlmsComponents.LLM_Components.AgentPipeline.agentdatacollection.dataloaderandpreprocess import AgentDataset, AgentDatasetLoader
from LlmsComponents.LLM_Components.AgentPipeline.agentdataretrieval.agentcontentretrieve import AgentContentRetrieval
from LlmsComponents.LLM_Components.AgentPipeline.agenthistory.agenthistorysession import AgentHistorySession
from LlmsComponents.LLM_Components.AgentPipeline.agenthistory.testinghistoryagentcontentretrieve import AgentHistoryManagerContentRetrieve
from LlmsComponents.LLM_Components.AgentPipeline.agentprompttemplate.agentprompting import AgentPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from LlmsComponents.LLM_Components.AgentPipeline.agentcallingpipline.agentchattemplete import AgentChatTokenizer


class AgentInteractionManager:
    def __init__(self, model_name: str, source_path: str, log_dir: str, model_kwargs: dict, encode_kwargs: dict,
                 dataset_names: List[str], multi_dataset_names: List[str]) -> None:
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.agent_retriever = AgentContentRetrieval(embedding_model=self.embedding_model)
        self.agent_history = AgentHistoryManagerContentRetrieve(embedding_model=self.embedding_model)
        self.dataset = AgentDataset(source=source_path).load_data()
        self.dataset_loader = AgentDatasetLoader(self.dataset)
        self.log_dir = log_dir
        self.dataset_names = dataset_names
        self.multi_dataset_names = multi_dataset_names
        os.makedirs(self.log_dir, exist_ok=True)
    
    def load_data(self) -> None:
        """Load datasets for the agent."""
        page_content, metadata = self.dataset_loader.load_dataset(self.dataset_names)
        page_content_multi, metadata_multi = self.dataset_loader.load_multiple_datasets(self.multi_dataset_names)
        self.agent_retriever.add_documents(page_content_multi, metadata_multi)

    def retrieve_single(self, query: str, top_k: int = 1) -> List[Tuple[Any, float]]:
        """Retrieve single query from the agent retriever."""
        return self.agent_retriever.retrieve(query, top_k)

    def batch_retrieve(self, queries: List[str], top_k: int = 1) -> List[List[Tuple[Any, float]]]:
        """Batch retrieval of queries from the agent retriever."""
        return self.agent_retriever.batch_retrieve(queries, top_k)

    def query_history(self, query: str, page_content_multi: List[str], metadata_multi: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Retrieve the agent history based on query."""
        return self.agent_history.query(query, page_content_multi, metadata_multi)

    def generate_prompt_response(self, query: str, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a prompt response."""
        template = AgentPromptTemplate(
            role="Advanced AI Agents will use that information, content, metadata",
            content="Your aim is to explain details about Query:\n {query} by looking at details in Content:\n{content} and Metadata:\n {metadata}",
            parameters={
                "query": query,
                "content": content,
                "metadata": metadata
            },
            example="Responding to your query by looking into content.",
            constraints=[
                "It should look like generated content.",
                "Don't provide any harmful content.",
                "Response should be crisp."
            ],
            output_format="Here is the detailed response to the query..."
        )
        return template.format()

    def generate_agent_response(self, formatted_prompt: str, model_name: str) -> str:
        """Generate an agent response based on the prompt."""
        chat_tokenizer = AgentChatTokenizer(model_name, quantization=None, cache_dir=self.log_dir)
        return chat_tokenizer.generate_response(json.dumps(formatted_prompt, indent=2))

    def log_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Log interactions to a file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(self.log_dir, f"interaction_{timestamp}.json")
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            json.dump(interaction_data, log_file, ensure_ascii=False, indent=4)

    def interact(self, query: str) -> None:
        """Interact with the agent using a query and log the interaction."""
        results = self.retrieve_single(query)
        content = results[0][0].page_content if results else ''
        metadata = results[0][0].metadata if results else {}
        formatted_prompt = self.generate_prompt_response(query, content, metadata)
        agent_response = self.generate_agent_response(formatted_prompt, "meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        interaction_data = {
            "query": query,
            "results": [{"content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in results],
            "prompt": formatted_prompt,
            "agent_response": agent_response
        }
        self.log_interaction(interaction_data)
        print(agent_response)





class AgentInteractionManager_testing:
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config['model_name']
        self.source_path = config['source_path']
        self.log_dir = config['log_dir']
        self.model_kwargs = config['model_kwargs']
        self.encode_kwargs = config['encode_kwargs']
        self.datasets = config['datasets']
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name, model_kwargs=self.model_kwargs, encode_kwargs=self.encode_kwargs
        )
        self.agent_retriever = AgentContentRetrieval(embedding_model=self.embedding_model)
        self.agent_history = AgentHistoryManagerContentRetrieve(embedding_model=self.embedding_model)
        self.dataset = AgentDataset(source=self.source_path).load_data()
        self.dataset_loader = AgentDatasetLoader(self.dataset)
        os.makedirs(self.log_dir, exist_ok=True)

    def load_data(self) -> None:
        page_content, metadata = self.dataset_loader.load_dataset(self.datasets['single'])
        page_content_multi, metadata_multi = self.dataset_loader.load_multiple_datasets(self.datasets['multiple'])
        self.agent_retriever.add_documents(page_content_multi, metadata_multi)

    def retrieve_single(self, query: str, top_k: int = 1) -> List[Tuple[Any, float]]:
        return self.agent_retriever.retrieve(query, top_k)

    def batch_retrieve(self, queries: List[str], top_k: int = 1) -> List[List[Tuple[Any, float]]]:
        return self.agent_retriever.batch_retrieve(queries, top_k)

    def query_history(self, query: str, page_content_multi: List[str], metadata_multi: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        return self.agent_history.query(query, page_content_multi, metadata_multi)

    def generate_prompt_response(self, query: str, content: str, metadata: Dict[str, Any]) -> str:
        template = AgentPromptTemplate(
            role="Advanced AI Agents will use that information, content, metadata",
            content="Your aim is to explain details about Query:\n {query} by looking at details in Content:\n{content} and Metadata:\n {metadata}",
            parameters={
                "query": query,
                "content": content,
                "metadata": metadata
            },
            example="Responding to your query by looking into content.",
            constraints=[
                "It should look like generated content.",
                "Don't provide any harmful content.",
                "Response should be crisp."
            ],
            output_format="Here is the detailed response to the query..."
        )
        return template.format()

    def generate_agent_response(self, formatted_prompt: str, model_name: str) -> str:
        chat_tokenizer = AgentChatTokenizer(model_name, quantization=None, cache_dir=self.log_dir)
        return chat_tokenizer.generate_response(json.dumps(formatted_prompt, indent=2))

    def log_interaction(self, interaction_data: Dict[str, Any]) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(self.log_dir, f"interaction_{timestamp}.json")
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            json.dump(interaction_data, log_file, ensure_ascii=False, indent=4)

    def interact(self, query: str) -> None:
        results = self.retrieve_single(query)
        content = results[0][0].page_content if results else ''
        metadata = results[0][0].metadata if results else {}
        formatted_prompt = self.generate_prompt_response(query, content, metadata)
        agent_response = self.generate_agent_response(formatted_prompt, "meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        interaction_data = {
            "query": query,
            "results": [{"content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in results],
            "prompt": formatted_prompt,
            "agent_response": agent_response
        }
        self.log_interaction(interaction_data)
        print(agent_response)


# def main() -> None:
#     with open('config.json', 'r') as config_file:
#         config = json.load(config_file)

#     agent_manager = AgentInteractionManager(config)
#     agent_manager.load_data()
#     agent_manager.interact("Linux terminal")

#     queries = [
#         "Career Counselor",
#         "Synonym finder",
#         "Commit Message Generator"
#     ]
#     for query in queries:
#         agent_manager.interact(query)


# if __name__ == "__main__":
#     main()





def main() -> None:
    """Main function to run the agent interaction manager."""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    source_path = r"C:\Users\heman\Desktop\components\output"
    log_dir = r"C:\Users\heman\Desktop\logs"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    dataset_names = ['prompts']
    multi_dataset_names = ['prompts', 'read', 'testing']
    
    agent_manager = AgentInteractionManager(model_name, source_path, log_dir, model_kwargs, encode_kwargs, dataset_names, multi_dataset_names)
    agent_manager.load_data()
    agent_manager.interact("Linux terminal")

    queries = [
        "Career Counselor",
        "Synonym finder",
        "Commit Message Generator"
    ]
    for query in queries:
        agent_manager.interact(query)


if __name__ == "__main__":
    main()