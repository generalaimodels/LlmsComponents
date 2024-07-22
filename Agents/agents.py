import json
import os
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import networkx as nx
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
import torch

# Constants
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MARKDOWN_SEPARATORS = ["\n\n", "\n", " ", ""]
CACHE_DIR = "/scratch/hemanth/LLMs/"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedAgent:
    """
    An advanced agent for processing and answering questions based on a knowledge base.
    """

    def __init__(self, knowledge_base_path: str, max_workers: int = 4):
        """
        Initialize the AdvancedAgent.

        Args:
            knowledge_base_path (str): Path to the knowledge base dataset.
            max_workers (int): Maximum number of worker threads.
        """
        try:
            self.raw_knowledge_base = self._load_knowledge_base(knowledge_base_path)
            self.docs_processed = self._split_documents(1024, self.raw_knowledge_base)
            self.vector_db = self._create_vector_database(self.docs_processed)
            self.reader_llm = self._create_reader_llm()
            self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME, cache_dir=CACHE_DIR)
            self.history: List[Dict[str, str]] = []
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        except Exception as e:
            logger.error(f"Error initializing AdvancedAgent: {e}")
            raise

    @staticmethod
    def _load_knowledge_base(dataset_path: str) -> List[LangchainDocument]:
        """
        Load the knowledge base from a dataset.

        Args:
            dataset_path (str): Path to the dataset.

        Returns:
            List[LangchainDocument]: List of loaded documents.
        """
        try:
            logger.info(f"Loading knowledge base from {dataset_path}")
            ds = load_dataset(dataset_path)
            return [
                LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
                for doc in tqdm(ds)
            ]
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise

    @staticmethod
    def _split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks.

        Args:
            chunk_size (int): Size of each chunk.
            knowledge_base (List[LangchainDocument]): List of documents to split.
            tokenizer_name (Optional[str]): Name of the tokenizer to use.

        Returns:
            List[LangchainDocument]: List of split documents.
        """
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
            logger.error(f"Error splitting documents: {e}")
            raise

    @staticmethod
    def _create_vector_database(docs_processed: List[LangchainDocument]) -> FAISS:
        """
        Create a vector database from processed documents.

        Args:
            docs_processed (List[LangchainDocument]): List of processed documents.

        Returns:
            FAISS: Vector database.
        """
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
            logger.error(f"Error creating vector database: {e}")
            raise

    @staticmethod
    def _create_reader_llm() -> pipeline:
        """
        Create a reader LLM pipeline.

        Returns:
            pipeline: Reader LLM pipeline.
        """
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
            logger.error(f"Error creating reader LLM: {e}")
            raise

    def process_batch(self, batch: Dict[str, List[str]]) -> List[str]:
        """
        Process a batch of inputs and return a list of responses.

        Args:
            batch (Dict[str, List[str]]): Batch of inputs.

        Returns:
            List[str]: List of responses.
        """
        try:
            futures = []
            for role, prompts in batch.items():
                for prompt in prompts:
                    futures.append(self.executor.submit(self.process_single_input, role, prompt))
            return [future.result() for future in futures]
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def process_single_input(self, role: str, prompt: str) -> str:
        """
        Process a single input and return a response.

        Args:
            role (str): Role of the input.
            prompt (str): Input prompt.

        Returns:
            str: Generated response.
        """
        try:
            retrieved_docs = self.vector_db.similarity_search(query=prompt, k=5)
            context = self._prepare_context(retrieved_docs)
            final_prompt = self._prepare_final_prompt(role, prompt, context)
            answer = self.reader_llm(final_prompt)[0]["generated_text"]
            self._log_interaction(role, prompt, answer)
            return answer
        except Exception as e:
            logger.error(f"Error processing single input: {e}")
            raise

    @staticmethod
    def _prepare_context(retrieved_docs: List[LangchainDocument]) -> str:
        """
        Prepare the context from retrieved documents.

        Args:
            retrieved_docs (List[LangchainDocument]): List of retrieved documents.

        Returns:
            str: Prepared context.
        """
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)]
        )
        return context

    def _prepare_final_prompt(self, role: str, prompt: str, context: str) -> str:
        """
        Prepare the final prompt for the reader LLM.

        Args:
            role (str): Role of the input.
            prompt (str): Input prompt.
            context (str): Prepared context.

        Returns:
            str: Final prompt for the reader LLM.
        """
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
        """
        Log the interaction to the history.

        Args:
            role (str): Role of the input.
            prompt (str): Input prompt.
            response (str): Generated response.
        """
        self.history.append({"role": role, "prompt": prompt, "response": response})

    def save_logs(self, directory: str) -> None:
        """
        Save the logs to a JSON file in the specified directory.

        Args:
            directory (str): Directory to save the logs.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, "interaction_logs.json")
            with open(file_path, "w") as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Logs saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
            raise

    @lru_cache(maxsize=1000)
    def cached_similarity_search(self, query: str, k: int = 5) -> List[LangchainDocument]:
        """
        Perform similarity search with caching for improved performance.

        Args:
            query (str): Query string.
            k (int): Number of similar documents to retrieve.

        Returns:
            List[LangchainDocument]: List of similar documents.
        """
        try:
            return self.vector_db.similarity_search(query=query, k=k)
        except Exception as e:
            logger.error(f"Error in cached similarity search: {e}")
            raise

    def update_knowledge_base(self, new_documents: List[Dict[str, str]]) -> None:
        """
        Update the knowledge base with new documents.

        Args:
            new_documents (List[Dict[str, str]]): List of new documents to add.
        """
        try:
            new_langchain_docs = [
                LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
                for doc in new_documents
            ]
            processed_new_docs = self._split_documents(1024, new_langchain_docs)
            self.vector_db.add_documents(processed_new_docs)
            logger.info(f"Added {len(processed_new_docs)} new documents to the knowledge base")
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            raise

    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """
        Generate a summary of the given text.

        Args:
            text (str): Text to summarize.
            max_length (int): Maximum length of the summary.

        Returns:
            str: Generated summary.
        """
        try:
            summary_prompt = f"Summarize the following text in no more than {max_length} words:\n\n{text}"
            summary = self.reader_llm(summary_prompt)[0]["generated_text"]
            return summary[:max_length]
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def answer_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Answer a question and provide sources for the information.

        Args:
            question (str): Question to answer.

        Returns:
            Dict[str, Any]: Dictionary containing the question, answer, and sources.
        """
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
            logger.error(f"Error answering with sources: {e}")
            raise

    def batch_process_async(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a batch of questions asynchronously.

        Args:
            batch (List[Dict[str, str]]): List of questions to process.

        Returns:
            List[Dict[str, Any]]: List of processed results.
        """
        try:
            futures = [self.executor.submit(self.answer_with_sources, item["question"]) for item in batch]
            return [future.result() for future in futures]
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
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
                logger.error(f"Error in interactive mode: {e}")
                print("An error occurred. Please try again.")

    def evaluate_performance(self, test_set: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Evaluate the agent's performance on a test set.

        Args:
            test_set (List[Dict[str, str]]): List of test questions and expected answers.

        Returns:
            Dict[str, float]: Performance metrics.
        """
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
            logger.error(f"Error evaluating performance: {e}")
            raise

    def export_knowledge_graph(self, output_file: str) -> None:
        """
        Export the knowledge base as a knowledge graph.

        Args:
            output_file (str): The path to the output file where the graph will be saved.

        Raises:
            IOError: If there's an error writing to the output file.
            ValueError: If the knowledge base is empty.
        """
        try:
            if not self.docs_processed:
                raise ValueError("Knowledge base is empty. Nothing to export.")

            graph = nx.Graph()
            node_mapping = {}

            # Add nodes to the graph
            for i, doc in enumerate(self.docs_processed):
                node_id = f"node_{i}"
                node_mapping[doc] = node_id
                graph.add_node(node_id, label=doc.page_content[:300])

            # Create edges based on similarity
            edge_list = []
            for doc in tqdm(self.docs_processed, desc="Creating edges"):
                similar_docs = self.cached_similarity_search(doc.page_content, k=5)
                for similar_doc in similar_docs:
                    if similar_doc != doc:
                        edge_list.append((node_mapping[doc], node_mapping[similar_doc]))

            graph.add_edges_from(edge_list)

            # Convert the graph to a dictionary representation
            graph_dict = nx.node_link_data(graph)

            # Write the graph to a JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(graph_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Knowledge graph exported to {output_file}")

        except ValueError as ve:
            logger.error(f"Error exporting knowledge graph: {str(ve)}")
            raise

        except IOError as io_error:
            logger.error(f"IO Error while exporting knowledge graph: {str(io_error)}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error exporting knowledge graph: {str(e)}")
            raise

def main():
    try:
        # Initialize the agent
        agent = AdvancedAgent(knowledge_base_path="path/to/your/dataset")

        # Example 1: Process a single question
        question = "What is the capital of France?"
        answer = agent.process_single_input("user", question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        # Example 2: Batch process questions
        batch_questions = [
            {"role": "user", "question": "Who invented the telephone?"},
            {"role": "user", "question": "What is the largest planet in our solar system?"}
        ]
        batch_answers = agent.process_batch({"user": [q["question"] for q in batch_questions]})
        for q, a in zip(batch_questions, batch_answers):
            print(f"Question: {q['question']}")
            print(f"Answer: {a}")

        # Example 3: Answer with sources
        question_with_sources = "What are the main causes of climate change?"
        result = agent.answer_with_sources(question_with_sources)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print("Sources:")
        for source in result['sources']:
            print(f"- {source}")

        # Example 4: Update knowledge base
        new_documents = [
            {"text": "New information about AI advancements.", "source": "AI Journal 2023"},
            {"text": "Recent discoveries in quantum computing.", "source": "Quantum Physics Review"}
        ]
        agent.update_knowledge_base(new_documents)

        # Example 5: Generate summary
        text_to_summarize = "Long text about the history of computing..."
        summary = agent.generate_summary(text_to_summarize, max_length=50)
        print(f"Summary: {summary}")

        # Example 6: Run interactive mode
        agent.interactive_mode()

        # Example 7: Evaluate performance
        test_set = [
            {"question": "What is the speed of light?", "expected_answer": "299,792,458 meters per second"},
            {"question": "Who wrote 'Romeo and Juliet'?", "expected_answer": "William Shakespeare"}
        ]
        performance = agent.evaluate_performance(test_set)
        print(f"Performance: {performance}")

        # Example 8: Export knowledge graph
        agent.export_knowledge_graph("knowledge_graph.json")

        # Save logs
        agent.save_logs("logs")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()