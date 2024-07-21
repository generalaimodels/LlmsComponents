import json
import os
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import torch
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache

# Constants
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MARKDOWN_SEPARATORS = ["\n\n", "\n", " ", ""]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAgent:
    def __init__(self, knowledge_base_path: str, max_workers: int = 4):
        self.raw_knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.docs_processed = self._split_documents(512, self.raw_knowledge_base)
        self.vector_db = self._create_vector_database(self.docs_processed)
        self.reader_llm = self._create_reader_llm()
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
        self.history: List[Dict[str, str]] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @staticmethod
    def _load_knowledge_base(dataset_path: str) -> List[LangchainDocument]:
        """Load the knowledge base from a dataset."""
        logger.info(f"Loading knowledge base from {dataset_path}")
        ds = load_dataset(dataset_path)
        return [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
            for doc in tqdm(ds)
        ]

    @staticmethod
    def _split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        """Split documents into chunks."""
        logger.info("Splitting documents")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
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
        unique_texts = set()
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

        return docs_processed_unique

    @staticmethod
    def _create_vector_database(docs_processed: List[LangchainDocument]) -> FAISS:
        """Create a vector database from processed documents."""
        logger.info("Creating vector database")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        return FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

    @staticmethod
    def _create_reader_llm() -> pipeline:
        """Create a reader LLM pipeline."""
        logger.info("Creating reader LLM")
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

    def process_batch(self, batch: Dict[str, List[str]]) -> List[str]:
        """Process a batch of inputs and return a list of responses."""
        futures = []
        for role, prompts in batch.items():
            for prompt in prompts:
                futures.append(self.executor.submit(self.process_single_input, role, prompt))
        return [future.result() for future in futures]

    def process_single_input(self, role: str, prompt: str) -> str:
        """Process a single input and return a response."""
        retrieved_docs = self.vector_db.similarity_search(query=prompt, k=5)
        context = self._prepare_context(retrieved_docs)
        final_prompt = self._prepare_final_prompt(role, prompt, context)
        answer = self.reader_llm(final_prompt)[0]["generated_text"]
        self._log_interaction(role, prompt, answer)
        return answer

    @staticmethod
    def _prepare_context(retrieved_docs: List[LangchainDocument]) -> str:
        """Prepare the context from retrieved documents."""
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)]
        )
        return context

    def _prepare_final_prompt(self, role: str, prompt: str, context: str) -> str:
        """Prepare the final prompt for the reader LLM."""
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": role,
                "content": f"""Context:
{context}
---
Now here is the question you need to answer.

Question: {prompt}""",
            },
        ]
        return self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

    def _log_interaction(self, role: str, prompt: str, response: str) -> None:
        """Log the interaction to the history."""
        self.history.append({"role": role, "prompt": prompt, "response": response})

    def save_logs(self, directory: str) -> None:
        """Save the logs to a JSON file in the specified directory."""
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "interaction_logs.json")
        with open(file_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Logs saved to {file_path}")

    @lru_cache(maxsize=1000)
    def cached_similarity_search(self, query: str, k: int = 5) -> List[LangchainDocument]:
        """Perform similarity search with caching for improved performance."""
        return self.vector_db.similarity_search(query=query, k=k)

    def update_knowledge_base(self, new_documents: List[Dict[str, str]]) -> None:
        """Update the knowledge base with new documents."""
        new_langchain_docs = [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
            for doc in new_documents
        ]
        processed_new_docs = self._split_documents(512, new_langchain_docs)
        self.vector_db.add_documents(processed_new_docs)
        logger.info(f"Added {len(processed_new_docs)} new documents to the knowledge base")

    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the given text."""
        summary_prompt = f"Summarize the following text in no more than {max_length} words:\n\n{text}"
        summary = self.reader_llm(summary_prompt)[0]["generated_text"]
        return summary[:max_length]

    def answer_with_sources(self, question: str) -> Dict[str, Any]:
        """Answer a question and provide sources for the information."""
        retrieved_docs = self.cached_similarity_search(question)
        context = self._prepare_context(retrieved_docs)
        final_prompt = self._prepare_final_prompt("user", question, context)
        answer = self.reader_llm(final_prompt)[0]["generated_text"]
        
        sources = [doc.metadata["source"] for doc in retrieved_docs]
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    def batch_process_async(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process a batch of questions asynchronously."""
        futures = [self.executor.submit(self.answer_with_sources, item["question"]) for item in batch]
        return [future.result() for future in futures]

    def interactive_mode(self) -> None:
        """Run the agent in interactive mode."""
        print("Welcome to the interactive mode. Type 'exit' to quit.")
        while True:
            question = input("Enter your question: ")
            if question.lower() == 'exit':
                break
            response = self.answer_with_sources(question)
            print(f"Answer: {response['answer']}")
            print("Sources:")
            for source in response['sources']:
                print(f"- {source}")
            print()

    def evaluate_performance(self, test_set: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate the agent's performance on a test set."""
        correct = 0
        total = len(test_set)
        
        for item in test_set:
            response = self.process_single_input("user", item["question"])
            if response.lower() == item["expected_answer"].lower():
                correct += 1
        
        accuracy = correct / total
        return {"accuracy": accuracy, "total_questions": total, "correct_answers": correct}

    def export_knowledge_graph(self, output_file: str) -> None:
        """Export the knowledge base as a knowledge graph."""
        # This is a placeholder implementation
        # In a real scenario, you would create a graph representation of your knowledge base
        graph = {
            "nodes": [{"id": i, "label": doc.page_content[:50]} for i, doc in enumerate(self.docs_processed)],
            "edges": []
        }
        
        # Create edges based on similarity
        for i, doc in enumerate(self.docs_processed):
            similar_docs = self.cached_similarity_search(doc.page_content, k=3)
            for similar_doc in similar_docs:
                if similar_doc != doc:
                    j = self.docs_processed.index(similar_doc)
                    graph["edges"].append({"source": i, "target": j})
        
        with open(output_file, "w") as f:
            json.dump(graph, f, indent=2)
        
        logger.info(f"Knowledge graph exported to {output_file}")