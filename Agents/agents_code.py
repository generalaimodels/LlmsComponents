import json
import os
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache

import torch
from tqdm import tqdm
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Constants
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MARKDOWN_SEPARATORS = ["\n\n", "\n", " ", ""]
CACHE_DIR = "/scratch/hemanth/LLMs/"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedAgent:
    def __init__(self, knowledge_base_path: str, max_workers: int = 4):
        try:
            self.raw_knowledge_base = self._load_knowledge_base(knowledge_base_path)
            self.docs_processed = self._split_documents(1024, self.raw_knowledge_base)
            self.vector_db = self._create_vector_database(self.docs_processed)
            self.reader_llm = self._create_reader_llm()
            self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME, cache_dir=CACHE_DIR)
            self.history: List[Dict[str, str]] = []
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        except Exception as e:
            logger.error(f"Error initializing AdvancedAgent: {str(e)}")
            raise

    @staticmethod
    def _load_knowledge_base(dataset_path: str) -> List[LangchainDocument]:
        """Load the knowledge base from a dataset."""
        try:
            logger.info(f"Loading knowledge base from {dataset_path}")
            ds = load_dataset(dataset_path)
            return [
                LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
                for doc in tqdm(ds)
            ]
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise

    @staticmethod
    def _split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        """Split documents into chunks."""
        try:
            logger.info("Splitting documents")
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=CACHE_DIR),
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
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise

    @staticmethod
    def _create_vector_database(docs_processed: List[LangchainDocument]) -> FAISS:
        """Create a vector database from processed documents."""
        try:
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
        except Exception as e:
            logger.error(f"Error creating vector database: {str(e)}")
            raise

    @staticmethod
    def _create_reader_llm() -> pipeline:
        """Create a reader LLM pipeline."""
        try:
            logger.info("Creating reader LLM")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                READER_MODEL_NAME, quantization_config=bnb_config, cache_dir=CACHE_DIR
            )
            tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME, cache_dir=CACHE_DIR)

            return pipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=1024,
            )
        except Exception as e:
            logger.error(f"Error creating reader LLM: {str(e)}")
            raise

    def process_batch(self, batch: Dict[str, List[str]]) -> List[str]:
        """Process a batch of inputs and return a list of responses."""
        try:
            futures = []
            for role, prompts in batch.items():
                for prompt in prompts:
                    futures.append(self.executor.submit(self.process_single_input, role, prompt))
            return [future.result() for future in futures]
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    def process_single_input(self, role: str, prompt: str) -> str:
        """Process a single input and return a response."""
        try:
            retrieved_docs = self.vector_db.similarity_search(query=prompt, k=5)
            context = self._prepare_context(retrieved_docs)
            final_prompt = self._prepare_final_prompt(role, prompt, context)
            answer = self.reader_llm(final_prompt)[0]["generated_text"]
            self._log_interaction(role, prompt, answer)
            return answer
        except Exception as e:
            logger.error(f"Error processing single input: {str(e)}")
            raise

    def _prepare_final_prompt(self, role: str, prompt: str, context: str) -> str:
        """Prepare the final prompt for the reader LLM."""
        try:
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
        except Exception as e:
            logger.error(f"Error preparing final prompt: {str(e)}")
            raise

    def _log_interaction(self, role: str, prompt: str, response: str) -> None:
        """Log the interaction to the history."""
        try:
            self.history.append({"role": role, "prompt": prompt, "response": response})
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")

    def save_logs(self, directory: str) -> None:
        """Save the logs to a JSON file in the specified directory."""
        try:
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, "interaction_logs.json")
            with open(file_path, "w") as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Logs saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving logs: {str(e)}")

    @lru_cache(maxsize=1000)
    def cached_similarity_search(self, query: str, k: int = 5) -> List[LangchainDocument]:
        """Perform similarity search with caching for improved performance."""
        try:
            return self.vector_db.similarity_search(query=query, k=k)
        except Exception as e:
            logger.error(f"Error in cached similarity search: {str(e)}")
            raise

    def update_knowledge_base(self, new_documents: List[Dict[str, str]]) -> None:
        """Update the knowledge base with new documents."""
        try:
            new_langchain_docs = [
                LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
                for doc in new_documents
            ]
            processed_new_docs = self._split_documents(1024, new_langchain_docs)
            self.vector_db.add_documents(processed_new_docs)
            logger.info(f"Added {len(processed_new_docs)} new documents to the knowledge base")
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")

    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the given text."""
        try:
            summary_prompt = f"Summarize the following text in no more than {max_length} words:\n\n{text}"
            summary = self.reader_llm(summary_prompt)[0]["generated_text"]
            return summary[:max_length]
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    def answer_with_sources(self, question: str) -> Dict[str, Any]:
        """Answer a question and provide sources for the information."""
        try:
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
        except Exception as e:
            logger.error(f"Error answering with sources: {str(e)}")
            raise

    def batch_process_async(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process a batch of questions asynchronously."""
        try:
            futures = [self.executor.submit(self.answer_with_sources, item["question"]) for item in batch]
            return [future.result() for future in futures]
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def interactive_mode(self) -> None:
        """Run the agent in interactive mode."""
        print("Welcome to the interactive mode. Type 'exit' to quit.")
        while True:
            try:
                question = input("Enter your question: ")
                if question.lower() == 'exit':
                    break
                response = self.answer_with_sources(question)
                print(f"Answer: {response['answer']}")
                print("Sources:")
                for source in response['sources']:
                    print(f"- {source}")
                print()
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print("An error occurred. Please try again.")

    def evaluate_performance(self, test_set: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate the agent's performance on a test set."""
        try:
            correct = 0
            total = len(test_set)
            
            for item in test_set:
                response = self.process_single_input("user", item["question"])
                if response.lower() == item["expected_answer"].lower():
                    correct += 1
            
            accuracy = correct / total
            return {"accuracy": accuracy, "total_questions": total, "correct_answers": correct}
        except Exception as e:
            logger.error(f"Error evaluating performance: {str(e)}")
            raise

    def export_knowledge_graph(self, output_file: str) -> None:
        """Export the knowledge base as a knowledge graph."""
        try:
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
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {str(e)}")
            raise
    def get_model_info(self) -> Dict[str, str]:
         """Return information about the models used."""
         return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reader_model": READER_MODEL_NAME,
    }
 
def get_knowledge_base_stats(self) -> Dict[str, int]:
    """Return statistics about the knowledge base."""
    return {
        "total_documents": len(self.docs_processed),
        "vector_store_size": self.vector_db.index.ntotal,
    }



def main():
    # Initialize the agent
    knowledge_base_path = "path/to/your/dataset"  # Replace with actual path
    agent = AdvancedAgent(knowledge_base_path)

    # Print model information
    print("Model Information:")
    print(agent.get_model_info())

    # Print knowledge base statistics
    print("\nKnowledge Base Statistics:")
    print(agent.get_knowledge_base_stats())

    # Example: Process a single question
    question = "What is the capital of France?"
    answer = agent.process_single_input("user", question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

    # Example: Process a batch of questions
    batch_questions = [
        {"role": "user", "question": "Who wrote Romeo and Juliet?"},
        {"role": "user", "question": "What is the boiling point of water?"},
        {"role": "user", "question": "Who invented the telephone?"}
    ]
    batch_answers = agent.batch_process_async(batch_questions)
    print("\nBatch Processing Results:")
    for q, a in zip(batch_questions, batch_answers):
        print(f"Q: {q['question']}")
        print(f"A: {a['answer']}")
        print(f"Sources: {', '.join(a['sources'])}\n")

    # Example: Update knowledge base
    new_documents = [
        {"text": "The Eiffel Tower is 324 meters tall.", "source": "Paris Tourism Guide"},
        {"text": "Python was created by Guido van Rossum.", "source": "Python Documentation"}
    ]
    agent.update_knowledge_base(new_documents)

    # Example: Generate a summary
    text_to_summarize = """
    Artificial Intelligence (AI) is a rapidly growing field of computer science focused on creating 
    intelligent machines that work and react like humans. It encompasses various subfields such as 
    machine learning, natural language processing, computer vision, and robotics. AI has applications 
    in numerous industries, including healthcare, finance, transportation, and entertainment.
    """
    summary = agent.generate_summary(text_to_summarize, max_length=50)
    print("\nSummary of AI text:")
    print(summary)

    # Example: Evaluate performance
    test_set = [
        {"question": "What is the capital of Japan?", "expected_answer": "Tokyo"},
        {"question": "Who painted the Mona Lisa?", "expected_answer": "Leonardo da Vinci"},
        {"question": "What is the largest planet in our solar system?", "expected_answer": "Jupiter"}
    ]
    performance = agent.evaluate_performance(test_set)
    print("\nPerformance Evaluation:")
    print(f"Accuracy: {performance['accuracy']:.2f}")
    print(f"Total Questions: {performance['total_questions']}")
    print(f"Correct Answers: {performance['correct_answers']}")

    # Example: Export knowledge graph
    output_file = "knowledge_graph.json"
    agent.export_knowledge_graph(output_file)
    print(f"\nKnowledge graph exported to {output_file}")

    # Example: Save logs
    log_directory = "logs"
    agent.save_logs(log_directory)
    print(f"Interaction logs saved to {log_directory}")

    # Example: Run interactive mode
    print("\nEntering Interactive Mode:")
    agent.interactive_mode()

if __name__ == "__main__":
    main()
    






import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union, Sequence, Mapping

import pandas as pd
from datasets import load_dataset, DatasetDict, DownloadConfig, DownloadMode, Split, VerificationMode
from langchain.docstore.document import Document
from tqdm import tqdm

def load_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Load documents from various file formats in the specified folder.
    """
    documents = []

    for file_path in Path(folder_path).rglob("*"):
        if file_path.is_file():
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

def process_dataset(dataset_path: str) -> List[Document]:
    """
    Load and process a dataset, combining all columns into page_content.
    """
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
            raw_knowledge_base.append(Document(page_content=page_content, metadata=metadata))

        return raw_knowledge_base
    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")
        return []

# Example usage
if __name__ == "__main__":
    folder_path = "path/to/your/folder"
    dataset_path = "path/to/your/dataset"

    # Load documents from the folder
    documents = load_documents(folder_path)
    for doc in documents:
        print(doc)

    # Process dataset
    knowledge_base = process_dataset(dataset_path)
    for doc in knowledge_base:
        print(doc)