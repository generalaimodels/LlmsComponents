import os
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime
from agentdataset import AgentDataset, AgentDatasetLoader
from agentcontentretrieve import AgentContentRetrieval
from agenthistorysession import AgentHistorySession
from historyagentcontentretrieve import AgentHistoryManagerContentRetrieve
from agentprompting import AgentPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from agentchattemplete import AgentChatTokenizer

class AgentInteractionManager:
    def __init__(self, model_name: str, source_path: str, log_dir: str, model_kwargs: dict, encode_kwargs: dict):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.agent_retriever = AgentContentRetrieval(embedding_model=self.embedding_model)
        self.agent_history = AgentHistoryManagerContentRetrieve(embedding_model=self.embedding_model)
        self.dataset = AgentDataset(source=source_path).load_data()
        self.dataset_loader = AgentDatasetLoader(self.dataset)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def load_data(self) -> None:
        page_content, metadata = self.dataset_loader.load_dataset('prompts')
        page_content_multi, metadata_multi = self.dataset_loader.load_multiple_datasets(['prompts', 'read', 'testing'])
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


def main():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    source_path = r"C:\Users\heman\Desktop\components\output"
    log_dir = r"C:\Users\heman\Desktop\components\logs"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    agent_manager = AgentInteractionManager(model_name, source_path, log_dir, model_kwargs, encode_kwargs)
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


























import os
import json
from typing import List, Tuple, Dict, Any
from agentdataset import AgentDataset, AgentDatasetLoader
from agentcontentretrieve import AgentContentRetrieval
from agenthistorysession import AgentHistorySession
from historyagentcontentretrieve import AgentHistoryManagerContentRetrieve
from agentprompting import AgentPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from agentchattemplete import AgentChatTokenizer


def load_agent_dataset(source: str) -> AgentDatasetLoader:
    agent_dataset = AgentDataset(source=source).load_data()
    return AgentDatasetLoader(agent_dataset)


def initialize_retriever(model_name: str, model_kwargs: Dict[str, Any], encode_kwargs: Dict[str, Any],
                         page_content: List[str], metadata: List[Dict[str, Any]]) -> AgentContentRetrieval:
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    agent_retriever = AgentContentRetrieval(embedding_model=embedding_model)
    agent_retriever.add_documents(page_content, metadata)
    return agent_retriever


def get_retrieval_results(agent_retriever: AgentContentRetrieval, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
    return agent_retriever.retrieve(query, top_k)


def get_batch_retrieval_results(agent_retriever: AgentContentRetrieval, queries: List[str], top_k: int = 1) -> List[List[Tuple[str, float]]]:
    return agent_retriever.batch_retrieve(queries, top_k)


def get_history_results(agent_history: AgentHistoryManagerContentRetrieve, query: str, page_content: List[str], metadata: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    return agent_history.query(query, page_content, metadata)


def generate_agent_response(template: AgentPromptTemplate, formatter_params: Dict[str, Any],
                            model_name: str, quantization: Any, cache_dir: str) -> str:
    formatted_prompt = template.format(**formatter_params)
    agent_input = json.dumps(formatted_prompt, indent=2)
    chat_tokenizer = AgentChatTokenizer(model_name, quantization, cache_dir)
    return chat_tokenizer.generate_response(agent_input)


def main():
    source = r"C:\Users\heman\Desktop\components\output"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    dataset_loader = load_agent_dataset(source)
    page_content, metadata = dataset_loader.load_dataset('prompts')
    page_content_multi, metadata_multi = dataset_loader.load_multiple_datasets(['prompts', 'read', 'testing'])

    agent_retriever = initialize_retriever(model_name, model_kwargs, encode_kwargs, page_content_multi, metadata_multi)
    agent_history = AgentHistoryManagerContentRetrieve(agent_retriever.embedding_model)

    query = "Linux terminal"
    results = get_retrieval_results(agent_retriever, query)
    print(f"\nQuery: {query}")
    for doc, score in results:
        print(f"Document: {doc.page_content}\nScore: {score}\nMetadata: {doc.metadata}\n")

    queries = ["Career Counselor", "Synonym finder", "Commit Message Generator"]
    batch_results = get_batch_retrieval_results(agent_retriever, queries)
    for i, query_results in enumerate(batch_results):
        print(f"\nQuery: {queries[i]}")
        for doc, score in query_results:
            print(f"Document: {doc.page_content}\nScore: {score}\nMetadata: {doc.metadata}\n")

    query = "Product Manager"
    results = get_history_results(agent_history, query, page_content_multi, metadata_multi)
    for doc, score in results:
        print(f"Document: {doc.page_content}\nMetadata: {doc.metadata}\nSimilarity Score: {score}\n---")

    template = AgentPromptTemplate(
        role="Advanced AI Agents will use that information, content, metadata",
        content="Your aim is to explain details about Query:\n {query} by looking at details in Content:\n{content} and Metadata:\n {metadata}",
        parameters={
            "query": query,
            "content": results[0][0].page_content,
            "metadata": results[0][0].metadata
        },
        example="Responding to your query by looking into content.",
        constraints=[
            "It should look like generated content.",
            "Don't provide any harmful content.",
            "Response should be crisp."
        ],
        output_format="Here is the detailed response to the query..."
    )

    formatter_params = {
        "query": query,
        "content": results[0][0].page_content,
        "metadata": results[0][0].metadata
    }
    response = generate_agent_response(
        template,
        formatter_params,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        quantization=None,
        cache_dir=r"C:\Users\heman\Desktop\components\model"
    )
    print(response)


if __name__ == '__main__':
    main()