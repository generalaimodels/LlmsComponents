# app.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from gradio.components import Component

from ragsystemspeed import (
    load_documents,
    split_documents,
    create_embeddings,
    save_vector_store,
    load_vector_store,
    query_vector_store,
    setup_logger,
)
from llmengine import get_llm_response, generate_prompt

# Set up logging
logger = setup_logger()

def load_or_create_vector_store(
    txt_output_dir: Path, index_path: Path, metadata_path: Path
) -> Any:
    """
    Loads an existing vector store if available; otherwise, creates a new one.

    Args:
        txt_output_dir (Path): Path to the directory containing text files.
        index_path (Path): Path to the index file.
        metadata_path (Path): Path to the metadata file.

    Returns:
        Any: The loaded or newly created vector store.

    Raises:
        FileNotFoundError: If no documents are found in the specified directory.
        ValueError: If no split documents are created from the documents.
    """
    try:
        if index_path.exists() and metadata_path.exists():
            vector_store = load_vector_store(index_path, metadata_path)
            logger.info("Loaded existing vector store.")
        else:
            documents = load_documents(txt_output_dir)
            if not documents:
                raise FileNotFoundError("No documents found in the specified directory.")
            
            split_docs = split_documents(documents)
            if not split_docs:
                raise ValueError("No split documents were created from the documents.")
            
            logger.info(f"Loaded {len(documents)} documents and split into {len(split_docs)} chunks")
            vector_store = create_embeddings(split_docs)
            save_vector_store(vector_store, index_path, metadata_path)
            logger.info("Created and saved new vector store.")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error in load_or_create_vector_store: {str(e)}")
        raise

def process_query(user_query: str, vector_store: Any, top_k: int = 5) -> Dict[str, Any]:
    """
    Processes the user query by querying the vector store and getting an LLM response.

    Args:
        user_query (str): The user's query.
        vector_store (Any): The vector store object.
        top_k (int): The number of top results to return.

    Returns:
        Dict[str, Any]: A dictionary containing the results and LLM response.
    """
    try:
        results = query_vector_store(vector_store, user_query, k=top_k)
        prompt = generate_prompt(user_query, results)
        response = get_llm_response(model="gpt-4o-mini", user_input=prompt)
        return {"results": results, "response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "results": [],
            "response": "An error occurred while processing your query. Please try again.",
        }

def format_response(response: str) -> str:
    """
    Formats the LLM response for better readability.

    Args:
        response (str): The raw LLM response.

    Returns:
        str: Formatted HTML string of the response.
    """
    paragraphs = response.split('\n\n')
    formatted_response = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f'<div class="response-content">{formatted_response}</div>'

def format_results(results: List[Dict[str, Any]]) -> str:
    """
    Formats the search results for display.

    Args:
        results (List[Dict[str, Any]]): List of search results.

    Returns:
        str: Formatted HTML string of the results.
    """
    formatted_results = ""
    for i, result in enumerate(results, 1):
        content = result.get("content", "").strip()
        metadata = result.get("metadata", {})
        formatted_results += (
            f'<div class="result-item">'
            f'<h4>Result {i}:</h4>'
            f'<div class="content"><strong>Content:</strong><br>{content}</div>'
            f'<div class="metadata"><strong>Metadata:</strong><br>{metadata}</div>'
            f'</div>'
        )
    return formatted_results

def gradio_interface(user_query: str) -> Tuple[str, str]:
    """
    Gradio interface function to process user query and return results.

    Args:
        user_query (str): The user's query.

    Returns:
        Tuple[str, str]: Formatted response and results.
    """
    try:
        output = process_query(user_query, vector_store)
        response = format_response(output["response"])
        results = format_results(output["results"])
        return response, results
    except Exception as e:
        logger.error(f"Error in Gradio interface: {str(e)}")
        error_message = (
            "<p>An error occurred while processing your query. Please try again later.</p>"
        )
        return error_message, "<p>No results available.</p>"


def create_gradio_blocks() -> gr.Blocks:
    """
    Creates and configures the Gradio interface.

    Returns:
        gr.Blocks: Configured Gradio Blocks interface.
    """
    css = """
   .container { max-width: 800px; margin: auto; padding: 20px; }
   #title { font-size: 32px; font-weight: bold; text-align: center; color: #333; margin-bottom: 20px; }
   #description { font-size: 18px; text-align: center; color: #666; margin-bottom: 30px; }
   .input-container { margin-bottom: 20px; }
   .output-container { margin-top: 20px; }
   .response-content { background-color: #f0f8ff; border: 1px solid #007ACC; padding: 15px; border-radius: 5px; }
   .result-item { background-color: #e9ffe9; border: 1px solid #28a745; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
   .result-item h4 { margin: 0 0 10px; color: #007ACC; }
   .content, .metadata { margin-bottom: 10px; }
    """

    with gr.Blocks(css=css, title="Document Search and LLM Query System") as demo:
        gr.HTML('<div class="container">')
        gr.HTML('<div id="title">Document Search and LLM Query System</div>')
        gr.HTML(
            '<div id="description">Enter your query below to receive a response and relevant content.</div>'
        )

        with gr.Row(elem_classes="input-container"):
            user_query = gr.Textbox(
                lines=2,
                placeholder="Enter your query here...",
                label="Your Query",
            )
        
        submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Row(elem_classes="output-container"):
            response = gr.HTML(label="Response")
        with gr.Row(elem_classes="output-container"):
            results = gr.HTML(label="Relevant Content")

        submit_btn.click(
            fn=gradio_interface,
            inputs=user_query,
            outputs=[response, results]
        )

        gr.HTML('</div>')  # Close container div

    return demo
# Main function to load vector store and launch Gradio interface
def main():
    """
    Main function to load vector store and launch Gradio interface.
    """
    # Define paths
    txt_output_dir = Path(r"E:\LLMS\Fine-tuning\output1")  # Update with your directory
    index_path = Path("faiss_index")
    metadata_path = Path("faiss_index_metadata")

    # Load or create vector store
    global vector_store
    try:
        vector_store = load_or_create_vector_store(
            txt_output_dir=txt_output_dir,
            index_path=index_path,
            metadata_path=metadata_path,
        )
    except Exception as e:
        logger.error(f"Failed to load or create vector store: {str(e)}")
        raise SystemExit("Unable to initialize vector store. Exiting application.")

    # Create Gradio app
    demo = create_gradio_blocks()

    # Check Gradio version and launch accordingly
    # gradio_version = version.parse(gr.__version__)
    # launch_kwargs = {
    #     "server_name": "0.0.0.0",
    #     "server_port": 7860,
    #     "share": True,
    # }

    # if gradio_version >= version.parse("3.0.0"):
    #     launch_kwargs["queue"] = True
    # else:
    #     logger.warning("Gradio version does not support queue. Running without queue enabled.")

    # Launch Gradio app
    demo.launch(share=True, inbrowser=True)

if __name__ == "__main__":
    main()