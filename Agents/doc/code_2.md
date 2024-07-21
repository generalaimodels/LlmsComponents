import os
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from diffusers import StableDiffusionPipeline

# Constants
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
CHUNK_SIZE = 512
LOG_DIR = "logs"

class Agent:
    def __init__(self, name: str):
        self.name = name
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{LOG_DIR}/{self.name}.txt")
        logger.addHandler(file_handler)
        return logger

    def process_input(self, text: str, image: Optional[str] = None, prompts: Optional[Dict[str, List[str]]] = None):
        raise NotImplementedError("This method should be implemented by subclasses")

class CodingAgent(Agent):
    def __init__(self):
        super().__init__("CodingAgent")
        self.model = AutoModel.from_pretrained("codegen-model")  # Replace with actual model

    def process_input(self, text: str, image: Optional[str] = None, prompts: Optional[Dict[str, List[str]]] = None):
        # Implement coding-specific processing
        result = self.model(text)
        self.logger.info(f"Processed coding input: {text}")
        return result

class ChatbotAgent(Agent):
    def __init__(self):
        super().__init__("ChatbotAgent")
        self.model = AutoModel.from_pretrained("chatbot-model")  # Replace with actual model

    def process_input(self, text: str, image: Optional[str] = None, prompts: Optional[Dict[str, List[str]]] = None):
        # Implement chatbot-specific processing
        result = self.model(text)
        self.logger.info(f"Processed chatbot input: {text}")
        return result

def split_documents(chunk_size: int, knowledge_base: List[LangchainDocument], tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME) -> List[LangchainDocument]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
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

def create_knowledge_base(dataset_path: str, target_column: str) -> List[LangchainDocument]:
    ds = load_dataset(dataset_path)
    return [
        LangchainDocument(page_content=doc[target_column], metadata={"second_column": doc["second_column"]})
        for doc in tqdm(ds)
    ]

def create_vector_database(docs_processed: List[LangchainDocument]) -> FAISS:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)

def main():
    # Initialize agents
    coding_agent = CodingAgent()
    chatbot_agent = ChatbotAgent()

    # Create knowledge base
    raw_knowledge_base = create_knowledge_base("path/to/dataset", "target_column")
    docs_processed = split_documents(CHUNK_SIZE, raw_knowledge_base)
    knowledge_vector_database = create_vector_database(docs_processed)

    # Example usage
    user_query = "How to implement a binary search tree?"
    retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=5)

    coding_result = coding_agent.process_input(user_query)
    chatbot_result = chatbot_agent.process_input(user_query)

    print(f"Coding Agent Result: {coding_result}")
    print(f"Chatbot Agent Result: {chatbot_result}")

if __name__ == "__main__":
    main()

To accomplish the task of creating agents (a coding agent and a generalized chatbot agent) that can handle text, images, and prompts using the transformers and Diffusers modules, and log history in separate directories, here’s a structured Python code snippet following PEP-8 standards and utilizing proper modules for typing and robustness.

### Library Imports

```python
import os
from typing import List, Dict, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
import torch

# Ensure necessary directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/coding_agent", exist_ok=True)
os.makedirs("logs/chatbot_agent", exist_ok=True)
```

### Agent Classes

```python
class BaseAgent:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def log_history(self, log: str):
        log_file = os.path.join(self.log_dir, "log.txt")
        with open(log_file, "a") as file:
            file.write(log + "\n")

    def process_batch(
        self, batch: Dict[str, Union[List[str], List[Dict[str, str]]]]
    ) -> List[str]:
        raise NotImplementedError("This method should be overridden by subclasses")
```

### CodingAgent Class

```python
class CodingAgent(BaseAgent):
    def __init__(self, log_dir: str, model_name: str):
        super().__init__(log_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def process_batch(
        self, batch: Dict[str, Union[List[str], List[Dict[str, str]]]]
    ) -> List[str]:
        responses = []
        for prompt in batch["prompts"]:
            response = self.pipeline(prompt)
            response_text = response[0]["generated_text"]
            responses.append(response_text)
            self.log_history(f"Prompt: {prompt}\nResponse: {response_text}\n")
        return responses
```

### ChatbotAgent Class

```python
class ChatbotAgent(BaseAgent):
    def __init__(self, log_dir: str, model_name: str):
        super().__init__(log_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline("conversational", model=self.model, tokenizer=self.tokenizer)

    def process_batch(
        self, batch: Dict[str, Union[List[str], List[Dict[str, str]]]]
    ) -> List[str]:
        responses = []
        for message in batch["messages"]:
            response = self.pipeline(message)
            response_text = response["generated_text"]
            responses.append(response_text)
            self.log_history(f"Message: {message}\nResponse: {response_text}\n")
        return responses
```

### Diffusion Image Generation

```python
class ImageGenerator:
    def __init__(self, model_name: str):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name)
        self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_images(self, prompts: List[str]) -> List[str]:
        images = []
        for prompt in prompts:
            image = self.pipeline(prompt).images[0]
            image_path = f"generated_images/{prompt.replace(' ', '_')}.png"
            image.save(image_path)
            images.append(image_path)
        return images
```

### Example Usage

```python
if __name__ == "__main__":
    coding_agent = CodingAgent(log_dir="logs/coding_agent", model_name="facebook/bart-large-cnn")
    chatbot_agent = ChatbotAgent(log_dir="logs/chatbot_agent", model_name="microsoft/DialoGPT-large")
    image_generator = ImageGenerator(model_name="CompVis/stable-diffusion-v-1-4")

    # Example batch input
    coding_batch = {"prompts": ["Write a function to reverse a string in Python."]}
    chatbot_batch = {"messages": ["Hello! How are you today?"]}

    coding_responses = coding_agent.process_batch(coding_batch)
    chatbot_responses = chatbot_agent.process_batch(chatbot_batch)
    image_prompts = ["A futuristic city skyline"]
    generated_images = image_generator.generate_images(image_prompts)

    print(f"Coding Responses: {coding_responses}")
    print(f"Chatbot Responses: {chatbot_responses}")
    print(f"Generated Images: {generated_images}")
```

### Explanation
1. **BaseAgent Class**:
   - A base class for shared functionalities, such as logging.
2. **CodingAgent Class**:
   - Processes textual prompts related to coding using a text-to-text generation pipeline.
3. **ChatbotAgent Class**:
   - Handles conversational inputs using a conversational pipeline.
4. **ImageGenerator Class**:
   - Utilizes Stable Diffusion to generate images based on textual prompts.
5. **Logging**:
   - Both `CodingAgent` and `ChatbotAgent` log their inputs and outputs to separate directories.

This structure ensures robust, optimized, and scalable code following PEP-8 standards and utilizing appropriate modules from the transformers and diffusers libraries.