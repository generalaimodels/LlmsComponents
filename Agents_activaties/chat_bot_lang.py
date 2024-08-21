import os
from typing import List, Tuple, Optional
from pathlib import Path
import re

import gradio as gr
import chromadb
from unidecode import unidecode

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory

# Constants
LLM_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-7b-it",
    "google/gemma-2b-it",
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-gemma-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mosaicml/mpt-7b-instruct",
    "tiiuae/falcon-7b-instruct",
    "google/flan-t5-xxl"
]
LLM_MODELS_SIMPLE = [os.path.basename(llm) for llm in LLM_MODELS]

def load_documents(file_paths: List[str], chunk_size: int, chunk_overlap: int) -> List:
    """Load PDF documents and create splits."""
    loaders = [PyPDFLoader(path) for path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(pages)

def create_vector_db(splits: List, collection_name: str) -> Chroma:
    """Create vector database from document splits."""
    embedding = HuggingFaceEmbeddings()
    client = chromadb.EphemeralClient()
    return Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=client,
        collection_name=collection_name
    )

def initialize_llm_chain(
    llm_model: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    vector_db: Chroma,
    progress: Optional[gr.Progress] = None
) -> ConversationalRetrievalChain:
    """Initialize LLM chain with specified parameters."""
    if progress:
        progress(0.1, desc="Initializing HF Hub...")

    llm_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "top_k": top_k,
    }

    if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm_kwargs["load_in_8bit"] = True
    elif llm_model in ["HuggingFaceH4/zephyr-7b-gemma-v0.1", "mosaicml/mpt-7b-instruct"]:
        raise gr.Error("LLM model is too large for free inference endpoint")
    elif llm_model == "microsoft/phi-2":
        llm_kwargs.update({"trust_remote_code": True, "torch_dtype": "auto"})
    elif llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        llm_kwargs["max_new_tokens"] = 250
    elif llm_model == "meta-llama/Llama-2-7b-chat-hf":
        raise gr.Error("Llama-2-7b-chat-hf model requires a Pro subscription")

    llm = HuggingFaceEndpoint(repo_id=llm_model, **llm_kwargs)

    if progress:
        progress(0.75, desc="Defining buffer memory...")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )

    if progress:
        progress(0.8, desc="Defining retrieval chain...")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_db.as_retriever(),
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    if progress:
        progress(0.9, desc="Done!")

    return qa_chain

def create_collection_name(filepath: str) -> str:
    """Generate a valid collection name from filepath."""
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ", "-")
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    
    if len(collection_name) < 3:
        collection_name += 'xyz'
    
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    
    return collection_name

def initialize_database(
    file_objs: List,
    chunk_size: int,
    chunk_overlap: int,
    progress: Optional[gr.Progress] = None
) -> Tuple[Chroma, str, str]:
    """Initialize the vector database from uploaded files."""
    file_paths = [obj.name for obj in file_objs if obj is not None]
    
    if progress:
        progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(file_paths[0])
    
    if progress:
        progress(0.25, desc="Loading documents...")
    doc_splits = load_documents(file_paths, chunk_size, chunk_overlap)
    
    if progress:
        progress(0.5, desc="Generating vector database...")
    vector_db = create_vector_db(doc_splits, collection_name)
    
    if progress:
        progress(0.9, desc="Done!")
    
    return vector_db, collection_name, "Complete!"

def initialize_llm(
    llm_option: int,
    llm_temperature: float,
    max_tokens: int,
    top_k: int,
    vector_db: Chroma,
    progress: Optional[gr.Progress] = None
) -> Tuple[ConversationalRetrievalChain, str]:
    """Initialize the LLM with specified parameters."""
    llm_name = LLM_MODELS[llm_option]
    qa_chain = initialize_llm_chain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"
def format_chat_history(message: str, chat_history: List[Tuple[str, str]]) -> List[str]:
    """Format chat history for the LLM."""
    formatted_history = []
    for user_message, bot_message in chat_history:
        formatted_history.append(f"User: {user_message}")
        formatted_history.append(f"Assistant: {bot_message}")
    return formatted_history

def conversation(
    qa_chain: ConversationalRetrievalChain,
    message: str,
    history: List[Tuple[str, str]]
) -> Tuple[ConversationalRetrievalChain, gr.update, List[Tuple[str, str]], str, int, str, int, str, int]:
    """Process a conversation turn and generate a response."""
    formatted_chat_history = format_chat_history(message, history)
    
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if "Helpful Answer:" in response_answer:
        response_answer = response_answer.split("Helpful Answer:")[-1].strip()
    
    response_sources = response["source_documents"]
    source_data = []
    for i in range(min(3, len(response_sources))):
        source = response_sources[i]
        source_data.append((
            source.page_content.strip(),
            source.metadata["page"] + 1
        ))
    
    while len(source_data) < 3:
        source_data.append(("", 0))
    
    new_history = history + [(message, response_answer)]
    
    return (qa_chain, gr.update(value=""), new_history,
            source_data[0][0], source_data[0][1],
            source_data[1][0], source_data[1][1],
            source_data[2][0], source_data[2][1])

def upload_file(file_objs: List[gr.File]) -> List[str]:
    """Process uploaded files and return their paths."""
    return [file.name for file in file_objs if file is not None]

def demo():
    """Create and launch the Gradio interface."""
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()
        
        gr.Markdown(
        """<center><h2>PDF-based chatbot</center></h2>
        <h3>Ask any questions about your PDF documents</h3>""")
        gr.Markdown(
        """<b>Note:</b> This AI assistant, using Langchain and open-source LLMs, performs RAG (Retrieval-Augmented Generation) on your PDF documents.""")

        with gr.Row():
            with gr.Column(scale=1):
                file_output = gr.File(file_count="multiple", label="Upload PDF files")
                chunk_size = gr.Slider(200, 1000, value=500, step=50, label="Chunk Size")
                chunk_overlap = gr.Slider(0, 100, value=50, step=10, label="Chunk Overlap")
                db_initialize = gr.Button("Initialize Database")
                db_status = gr.Textbox(label="Database Status", interactive=False)

            with gr.Column(scale=1):
                llm_option = gr.Dropdown(choices=LLM_MODELS_SIMPLE, value=LLM_MODELS_SIMPLE[0], label="LLM Model")
                llm_temperature = gr.Slider(0, 1, value=0.1, step=0.1, label="Temperature")
                max_tokens = gr.Slider(32, 4096, value=512, step=32, label="Max Tokens")
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top K")
                llm_initialize = gr.Button("Initialize LLM")
                llm_status = gr.Textbox(label="LLM Status", interactive=False)

        chatbot = gr.Chatbot(label="Conversation")
        msg = gr.Textbox(label="Your message")
        clear = gr.Button("Clear")

        with gr.Row():
            with gr.Column(scale=1):
                source1 = gr.Textbox(label="Source 1", interactive=False)
                page1 = gr.Number(label="Page 1", interactive=False)
            with gr.Column(scale=1):
                source2 = gr.Textbox(label="Source 2", interactive=False)
                page2 = gr.Number(label="Page 2", interactive=False)
            with gr.Column(scale=1):
                source3 = gr.Textbox(label="Source 3", interactive=False)
                page3 = gr.Number(label="Page 3", interactive=False)

        # Event handlers
        db_initialize.click(
            initialize_database,
            inputs=[file_output, chunk_size, chunk_overlap],
            outputs=[vector_db, collection_name, db_status]
        )

        llm_initialize.click(
            initialize_llm,
            inputs=[llm_option, llm_temperature, max_tokens, top_k, vector_db],
            outputs=[qa_chain, llm_status]
        )

        msg.submit(
            conversation,
            inputs=[qa_chain, msg, chatbot],
            outputs=[qa_chain, msg, chatbot, source1, page1, source2, page2, source3, page3]
        )

        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()

if __name__ == "__main__":
    demo()