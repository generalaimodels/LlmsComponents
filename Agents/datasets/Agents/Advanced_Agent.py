import json
import os
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
from datetime import datetime
import networkx as nx
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
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Constants
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
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
            dataset = load_dataset(dataset_path)
            dataset=dataset['train']
            return [
                LangchainDocument(page_content=str(doc["text"]), metadata={"source": doc["text"]})
                for doc in tqdm(dataset)
            ]
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
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
            tokenizer.pad_token=tokenizer.eos_token
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
        """Process a single input and return a response, considering conversation history."""
        try:
            # Retrieve relevant documents based on the entire conversation context
            conversation_context = self._get_conversation_context()
            retrieved_docs = self.vector_db.similarity_search(query=conversation_context + "\n" + prompt, k=5)
            context = self._prepare_context(retrieved_docs)
            
            # Prepare the final prompt with conversation history
            final_prompt = self._prepare_final_prompt(role, prompt, context, conversation_context)
            
            # Generate the response
            answer = self.reader_llm(final_prompt)[0]["generated_text"]
            
            # Log the interaction
            self._log_interaction(role, prompt, answer)
            
            return answer
        except Exception as e:
            logger.error(f"Error processing single input: {str(e)}")
            raise
    def _get_conversation_context(self, max_turns: int = 5) -> str:
        """Get the recent conversation context."""
        context = "  "
        for turn in self.history[-max_turns:]:
            context += f"{turn['role']}: {turn['prompt']}\n"
            context += f"Assistant: {turn['response']}\n"
        return context.strip()
    
    def _prepare_final_prompt(self, role: str, prompt: str, context: str, conversation_context: str) -> str:
        """Prepare the final prompt for the reader LLM, including conversation history."""
        try:
            system_message = f"""You are an AI assistant with access to a knowledge base. 
            Use the information in the context to answer questions comprehensively. 
            Consider the conversation history when formulating your response. 
            If you can't answer based on the given information, say so.{role}"""

            prompt_in_chat_format = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Conversation history:\n{conversation_context}"},
                {"role": "user", "content": f"Context from knowledge base:\n{context}"},
                {"role": "user", "content": f"New question: {prompt}"}
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
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "prompt": prompt,
                "response": response
            })
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
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.history = []

    def get_conversation_summary(self) -> str:
        """Generate a summary of the entire conversation."""
        try:
            conversation = self._get_conversation_context(max_turns=len(self.history))
            prompt = f"Summarize the following conversation:\n\n{conversation}\n\nSummary:"
            summary = self.reader_llm(prompt)[0]["generated_text"]
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            raise
    
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
            conversation_context = self._get_conversation_context()
            retrieved_docs = self.cached_similarity_search(question)
            context = self._prepare_context(retrieved_docs)
            final_prompt = self._prepare_final_prompt("user", question, context,conversation_context)
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
                question = str(input("Enter your question: "))
                if question.lower() == 'exit':
                    break
                response = self.answer_with_sources(question)
                # response=self.process_single_input(role="role", prompt=question)
                print(f"Answer: {response['answer']}")
                # print("Sources:")
                # for source in response['sources']:
                #     print(f"- {source}")
                # print()
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


    #model details logic is pending``
    def get_model_info(self) -> Dict[str, str]:
         """Return information about the models used."""
         return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reader_model": READER_MODEL_NAME,
    }
    #update logic update is pending
    def get_knowledge_base_stats(self) -> Dict[str, int]:
        """Return statistics about the knowledge base."""
        return {
        "total_documents": len(self.docs_processed),
        "vector_store_size": self.vector_db.index.ntotal,
    }
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
    
            # Add nodes to the graph with unique identifiers
            for i, doc in enumerate(self.docs_processed):
                node_id = f"node_{i}"
                node_mapping[i] = node_id  # Map document index to node ID
                graph.add_node(node_id, label=doc.page_content[:300])
    
            # Create edges based on similarity
            edge_list = []
            for i, doc in tqdm(enumerate(self.docs_processed), desc="Creating edges"):
                similar_docs = self.cached_similarity_search(doc.page_content, k=5)
                for similar_doc in similar_docs:
                    if similar_doc in self.docs_processed:
                        j = self.docs_processed.index(similar_doc)
                        if j != i:  # Avoid self-loops
                            edge_list.append((node_mapping[i], node_mapping[j]))
    
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
    # Initialize the agent
    knowledge_base_path = "/scratch/hemanth/LLMs/out_papers/reformatted"  # Replace with actual path
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
        # print(f"Sources: {', '.join(a['sources'])}\n")

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
    summary = agent.generate_summary(text_to_summarize, max_length=150)
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