import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Optional, Any

import torch
from datasets import load_dataset
from functools import lru_cache
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MARKDOWN_SEPARATORS = ["\n \n", "\n", " ", ""]

class AdvancedAgent:
    def __init__(
                 self,
                 knowledge_base_path: str,
                 log_folder: str = "logs",
                 max_workers: int = 4,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 reader_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 cache_dir: Optional[str] = None,
                 temperature: float = 0.2,
                 repetition_penalty: float = 1.1,
                 max_new_tokens: int = 500):
        self.log_folder = log_folder
        self.ensure_log_folder()
        self.history: List[Dict[str, Any]] = []

        try:
            self.raw_knowledge_base = self.load_and_process_dataset(knowledge_base_path)
            self.embedding_model = self.initialize_embedding_model(model_name=embedding_model)
            self.docs_processed = self.split_document_cus(1024, self.raw_knowledge_base)
            self.vector_db = self.create_knowledge_vector_database(self.docs_processed, embedding_model=self.embedding_model)
            self.reader_llm = self.create_reader_llm(reader_model_name=reader_model_name, cache_dir=cache_dir,
                                                     temperature=temperature, repetition_penalty=repetition_penalty,
                                                     max_new_tokens=max_new_tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(reader_model_name, cache_dir=cache_dir)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        except Exception as e:
            logger.error(f"Error initializing AdvancedAgent: {str(e)}")
            raise

    def ensure_log_folder(self) -> None:
        """Ensure the log folder exists."""
        os.makedirs(self.log_folder, exist_ok=True)

    def log_interaction(self, role: str, content: str, source: Optional[str] = None) -> None:
        """Log an interaction to a JSON file."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "role": role,
            "content": content,
            "source": source
        }
        
        filename = f"{timestamp.split('T')[0]}.json"
        filepath = os.path.join(self.log_folder, filename)
        
        with open(filepath, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
        
        self.history.append(log_entry)

    @staticmethod
    def load_and_process_dataset(dataset_path: str) -> List[LangchainDocument]:
        """Load and process a dataset, combining all columns into page_content."""
        try:
            ds = load_dataset(dataset_path)
            if 'train' not in ds:
                raise KeyError("Dataset does not contain a 'train' split.")
            
            ds = ds['train']
            columns = ds.column_names
    
            raw_knowledge_base = []
            for doc in tqdm(ds, desc="Processing documents"):
                page_content = "  ".join(str(doc[col]) for col in columns)
                metadata = {col: str(doc[col]) for col in columns}
                raw_knowledge_base.append(LangchainDocument(page_content=page_content, metadata=metadata))
    
            return raw_knowledge_base
        except Exception as e:
            logger.error(f"An error occurred while processing the dataset: {e}")
            return []

    def split_document_cus(self, chunk_size: int, knowledge_base: List[LangchainDocument]) -> List[LangchainDocument]:
        """Split documents into chunks of maximum size `chunk_size` tokens."""
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.embedding_model,
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
            content_hash = hash(doc.page_content)
            if content_hash not in unique_docs:
                unique_docs[content_hash] = doc
        return list(unique_docs.values())

    @staticmethod
    def create_knowledge_vector_database(docs_processed: List[LangchainDocument], embedding_model: HuggingFaceEmbeddings) -> FAISS:
        """Creates a knowledge vector database from processed documents."""
        try:
            return FAISS.from_documents(
                docs_processed, 
                embedding_model, 
                distance_strategy=DistanceStrategy.COSINE
            )
        except Exception as e:
            logger.error(f"Error creating knowledge vector database: {e}")
            raise

    @staticmethod
    def initialize_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
        """Initializes the HuggingFace embedding model."""
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                multi_process=True,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            return embedding_model
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    @staticmethod
    def create_reader_llm(reader_model_name: str, cache_dir: Optional[str] = None,
                          temperature: float = 0.2, repetition_penalty: float = 1.1,
                          max_new_tokens: int = 500):
        """Create a reader LLM pipeline using the specified model."""
        try:
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
        except Exception as e:
            logger.error(f"Error creating reader LLM: {e}")
            raise

    def prepare_context(self, retrieved_docs: List[LangchainDocument], max_context_length: int) -> str:
        """Prepare the context from retrieved documents."""
        context = "\nExtracted documents:\n"
        current_length = len(self.tokenizer.encode(context))
     
        for i, doc in enumerate(retrieved_docs):
            doc_text = f"Document {i}:::\n{doc.page_content}\n"
            doc_length = len(self.tokenizer.encode(doc_text))
            
            if current_length + doc_length > max_context_length:
                break
            
            context += doc_text
            current_length += doc_length
     
        return context

    def generate_answer(self, query: str, retrieved_docs: List[LangchainDocument],
                        system_prompt: str = "You are an advanced AI agent capable of answering questions based on provided context.",
                        max_context_length: int = 2048) -> str:
        """Generate an answer based on the query and retrieved documents."""
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
    
        rag_prompt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
    
        context = self.prepare_context(retrieved_docs, max_context_length)
        
        final_prompt = rag_prompt_template.format(question=query, context=context)
    
        answer = self.reader_llm(final_prompt)[0]["generated_text"]
        return answer

    @lru_cache(maxsize=1000)
    def cached_similarity_search(self, query: str, k: int = 5) -> List[LangchainDocument]:
        """Perform similarity search with caching for improved performance."""
        try:
            return self.vector_db.similarity_search(query=query, k=k)
        except Exception as e:
            logger.error(f"Error in cached similarity search: {str(e)}")
            raise

    def process_query(self, query: str) -> str:
        """Process a single query and generate a response."""
        try:
            retrieved_docs = self.cached_similarity_search(query)
            answer = self.generate_answer(query, retrieved_docs)
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

    def single_prompt_function(self, prompt: str) -> str:
        """Process a single prompt with history activation."""
        self.log_interaction("user", prompt)
        response = self.process_query(prompt)
        self.log_interaction("agent", response)
        return response

    def batch_size_function(self, prompts: List[str], batch_size: int = 5) -> List[str]:
        """Process a batch of prompts with history activation."""
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_responses = []
            for prompt in batch:
                response = self.single_prompt_function(prompt)
                batch_responses.append(response)
            responses.extend(batch_responses)
        return responses

    def interface_terminal(self) -> None:
        """Run an interactive terminal interface with history."""
        print("Welcome to the Advanced Agent Terminal Interface")
        print("Type 'exit' to leave the interface.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = self.single_prompt_function(user_input)
            print(f"Agent: {response}")

class HistoryManager:
    """Manages the query history and extracts relevant history based on the query."""
    
    def __init__(self, history: List[Dict[str, Any]]):
        self.history = history

    def extract_relevant_history(self, query: str, max_history_items: int = 5) -> List[Dict[str, Any]]:
        """Extract relevant history items related to the query."""
        relevant_history = []
        # Simple relevance check: looking for query terms in the content of the history items
        for item in self.history:
            if query.lower() in item['content'].lower():
                relevant_history.append(item)
            if len(relevant_history) >= max_history_items:
                break
        return relevant_history

    def log_history(self, role: str, content: str, source: Optional[str] = None) -> None:
        """Logs history entries."""
        timestamp = datetime.now().isoformat()
        self.history.append({
            "timestamp": timestamp,
            "role": role,
            "content": content,
            "source": source
        })


class AdvancedContextManager:
    """Manages the context for the queries, using relevant history and knowledge base documents."""
    
    def __init__(self, knowledge_base: List[LangchainDocument], tokenizer: Any):
        self.knowledge_base = knowledge_base
        self.tokenizer = tokenizer

    def prepare_context(self, query: str, relevant_history: List[Dict[str, Any]], retrieved_docs: List[LangchainDocument], max_context_length: int = 2048) -> str:
        """Prepare the context from relevant history and retrieved documents."""
        context = "Relevant History:\n"
        current_length = len(self.tokenizer.encode(context))

        for history_item in relevant_history:
            history_text = f"{history_item['role']}: {history_item['content']}\n"
            history_length = len(self.tokenizer.encode(history_text))

            if current_length + history_length > max_context_length:
                break

            context += history_text
            current_length += history_length

        context += "\nRetrieved Documents:\n"
        for i, doc in enumerate(retrieved_docs):
            doc_text = f"Document {i}:::\n{doc.page_content}\n"
            doc_length = len(self.tokenizer.encode(doc_text))
            
            if current_length + doc_length > max_context_length:
                break
            
            context += doc_text
            current_length += doc_length

        return context


# Example usage
if __name__ == "__main__":
    agent = AdvancedAgent(knowledge_base_path="path_to_dataset")
    history_manager = HistoryManager(agent.history)
    context_manager = AdvancedContextManager(agent.raw_knowledge_base, agent.tokenizer)

    def process_query_with_context(query: str) -> str:
        try:
            relevant_history = history_manager.extract_relevant_history(query)
            retrieved_docs = agent.cached_similarity_search(query)
            context = context_manager.prepare_context(query, relevant_history, retrieved_docs)

            answer = agent.generate_answer(query, retrieved_docs, system_prompt="You are an advanced AI agent capable of answering questions based on provided context.", max_context_length=2048)
            return answer
        except Exception as e:
            logger.error(f"Error processing query with context: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

    # Interactive terminal interface
    print("Welcome to the Enhanced Advanced Agent Interface")
    print("Type 'exit' to leave the interface.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = process_query_with_context(user_input)
        print(f"Agent: {response}")