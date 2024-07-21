import json
import os
from typing import List, Optional, Dict
from tqdm import tqdm
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import torch

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MARKDOWN_SEPARATORS = ["\n\n", "\n", " ", ""]


def load_knowledge_base(dataset_path: str) -> List[LangchainDocument]:
    """
    Load the knowledge base from a dataset.
    """
    ds = load_dataset(dataset_path)
    return [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(ds)
    ]


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
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
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def create_vector_database(docs_processed: List[LangchainDocument]) -> FAISS:
    """
    Create a vector database from processed documents.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )


def create_reader_llm() -> pipeline:
    """
    Create a reader LLM pipeline.
    """
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


class AdvancedAgent:
    def __init__(self, knowledge_base_path: str):
        self.raw_knowledge_base = load_knowledge_base(knowledge_base_path)
        self.docs_processed = split_documents(512, self.raw_knowledge_base)
        self.vector_db = create_vector_database(self.docs_processed)
        self.reader_llm = create_reader_llm()
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
        self.history = []

    def process_batch(self, batch: Dict[str, List[str]]) -> List[str]:
        """
        Process a batch of inputs and return a list of responses.
        """
        responses = []
        for role, prompts in batch.items():
            for prompt in prompts:
                response = self.process_single_input(role, prompt)
                responses.append(response)
        return responses

    def process_single_input(self, role: str, prompt: str) -> str:
        """
        Process a single input and return a response.
        """
        retrieved_docs = self.vector_db.similarity_search(query=prompt, k=5)
        context = self.prepare_context(retrieved_docs)
        final_prompt = self.prepare_final_prompt(role, prompt, context)
        answer = self.reader_llm(final_prompt)[0]["generated_text"]
        self.log_interaction(role, prompt, answer)
        return answer

    def prepare_context(self, retrieved_docs: List[LangchainDocument]) -> str:
        """
        Prepare the context from retrieved documents.
        """
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
        )
        return context

    def prepare_final_prompt(self, role: str, prompt: str, context: str) -> str:
        """
        Prepare the final prompt for the reader LLM.
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
        rag_prompt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
        return rag_prompt_template

    def log_interaction(self, role: str, prompt: str, response: str):
        """
        Log the interaction to the history.
        """
        self.history.append({"role": role, "prompt": prompt, "response": response})

    def save_logs(self, directory: str):
        """
        Save the logs to a JSON file in the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "interaction_logs.json")
        with open(file_path, "w") as f:
            json.dump(self.history, f, indent=2)


# Usage example
if __name__ == "__main__":
    agent = AdvancedAgent("path/to/your/dataset")
    batch = {
        "user": ["How to create a pipeline object?", "What is natural language processing?"],
        "system": ["Explain the concept of transfer learning."],
    }
    responses = agent.process_batch