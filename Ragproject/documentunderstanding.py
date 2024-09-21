import argparse
import logging
import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import faiss
import fitz  # PyMuPDF for PDF processing
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and lightweight model
CHUNK_SIZE: int = 512  # Adjust based on your needs
CPU_COUNT: int = multiprocessing.cpu_count()

# Load tokenizer and model once
try:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(DEVICE)
    MODEL.eval()
    logger.info(f"Model and tokenizer loaded on {DEVICE}.")
except Exception as e:
    logger.error(f"Error loading model {EMBEDDING_MODEL_NAME}: {e}")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using PyMuPDF.

    Parameters:
    pdf_path (Path): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF file.
    """
    try:
        text: str = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file {pdf_path}: {e}")
        return ""


def extract_text_from_txt(txt_path: Path) -> str:
    """
    Extract text from a TXT file.

    Parameters:
    txt_path (Path): Path to the TXT file.

    Returns:
    str: Content of the TXT file.
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading TXT file {txt_path}: {e}")
        return ""


def process_file(file_path: Path) -> Tuple[str, str]:
    """
    Reads a file and returns its text content along with the file name.

    Parameters:
    file_path (Path): Path to the file.

    Returns:
    Tuple[str, str]: Tuple containing the extracted text and the file path as a string.
    """
    ext: str = file_path.suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        return "", ""
    return text, str(file_path)


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Splits the text into chunks of a specified size.

    Parameters:
    text (str): The text to be chunked.
    chunk_size (int): The chunk size (number of tokens).

    Returns:
    List[str]: List of text chunks.
    """
    tokens: List[str] = TOKENIZER.tokenize(text)
    chunks: List[str] = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk = TOKENIZER.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the preloaded model.

    Parameters:
    texts (List[str]): List of text chunks.

    Returns:
    np.ndarray: NumPy array containing the embeddings.
    """
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for text in texts:
            inputs = TOKENIZER(
                text, return_tensors="pt", truncation=True, padding=True
            ).to(DEVICE)
            outputs = MODEL(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(embedding)
    embeddings = np.vstack(embeddings)
    return embeddings


def process_and_chunk_file(args: Tuple[Path, int]) -> List[Tuple[str, np.ndarray, str]]:
    """
    Processes a single file: extracts text, chunks it, and generates embeddings.
    Returns a list of tuples with (chunk_text, embedding, source_info).

    Parameters:
    args (Tuple[Path, int]): Tuple containing the file path and chunk size.

    Returns:
    List[Tuple[str, np.ndarray, str]]: List of tuples containing chunk text, embedding, and source.
    """
    file_path, chunk_size = args
    text, source = process_file(file_path)
    if not text:
        return []
    chunks = chunk_text(text, chunk_size)
    if not chunks:
        return []
    embeddings = embed_texts(chunks)
    results: List[Tuple[str, np.ndarray, str]] = []
    for chunk_text, embedding in zip(chunks, embeddings):
        results.append((chunk_text, embedding, source))
    return results


def build_vector_database(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a FAISS vector database from the embeddings.

    Parameters:
    embeddings (np.ndarray): NumPy array containing the embeddings.

    Returns:
    faiss.IndexFlatL2: The FAISS index.
    """
    if embeddings.size == 0:
        logger.error("No embeddings provided to build vector database.")
        raise ValueError("Embeddings array is empty.")
    dimension: int = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_vector_database(index: faiss.IndexFlatL2, path: Path) -> None:
    """
    Saves the FAISS index to the specified path.

    Parameters:
    index (faiss.IndexFlatL2): The FAISS index to save.
    path (Path): Path to save the index file.
    """
    try:
        faiss.write_index(index, str(path))
        logger.info(f"Vector database saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save vector database: {e}")


def load_vector_database(path: Path) -> faiss.IndexFlatL2:
    """
    Loads a FAISS index from the specified path.

    Parameters:
    path (Path): Path to the index file.

    Returns:
    faiss.IndexFlatL2: The loaded FAISS index.
    """
    try:
        index = faiss.read_index(str(path))
        logger.info(f"Vector database loaded from {path}")
        return index
    except Exception as e:
        logger.error(f"Failed to load vector database: {e}")
        raise


def search_vector_database(
    query: str, index: faiss.IndexFlatL2, k: int = 5
) -> List[Tuple[int, float]]:
    """
    Searches the vector database for the top k most similar chunks to the query.

    Parameters:
    query (str): The query string.
    index (faiss.IndexFlatL2): The FAISS index to search.
    k (int): Number of top results to return.

    Returns:
    List[Tuple[int, float]]: List of tuples containing the index of the chunk and its distance to the query.
    """
    query_embedding = embed_texts([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = list(zip(indices[0], distances[0]))
    return results


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="PDF/Text Document Similarity Search")

    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="Input file paths or directories containing PDF/TXT files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output"),
        help="Directory to save processed data and vector database.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query string to search the vector database.",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=5,
        help="Number of top similar results to retrieve.",
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="Build the vector database from input files.",
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=None,
        help="Path to the vector database index file.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main function to process files, build vector database, and handle queries.
    """
    try:
        args = parse_arguments()
        input_paths = args.input_paths
        output_dir = args.output_dir
        query = args.query
        k = args.top_k
        build_index = args.build_index
        index_path = args.index_path

        if not build_index and not query:
            logger.error(
                "No action specified. Use --build_index to build index or provide --query to search."
            )
            sys.exit(1)

        if build_index:
            # Collect all files
            files_to_process: List[Path] = []
            for path in input_paths:
                if path.is_file():
                    files_to_process.append(path)
                elif path.is_dir():
                    pdf_files = list(path.rglob("*.pdf"))
                    txt_files = list(path.rglob("*.txt"))
                    files_in_dir = pdf_files + txt_files
                    files_to_process.extend(files_in_dir)
                else:
                    logger.warning(f"Path {path} is neither a file nor a directory.")

            if not files_to_process:
                logger.error("No valid files to process.")
                sys.exit(1)

            pool = Pool(processes=CPU_COUNT)
            args_list = [(path, CHUNK_SIZE) for path in files_to_process]
            results: List[Tuple[str, np.ndarray, str]] = []

            logger.info("Starting file processing and embedding generation...")
            for res in tqdm(
                pool.imap_unordered(process_and_chunk_file, args_list),
                total=len(args_list),
                desc="Processing files",
            ):
                if res:
                    results.extend(res)
            pool.close()
            pool.join()

            if not results:
                logger.error("No results to process.")
                sys.exit(1)

            # Separate embeddings and sources
            embeddings = np.vstack([item[1] for item in results]).astype("float32")
            chunk_texts = [item[0] for item in results]
            sources = [item[2] for item in results]

            # Build vector database
            logger.info("Building vector database...")
            index = build_vector_database(embeddings)

            # Save vector database
            if index_path is None:
                index_path = output_dir / "vector_database.index"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            save_vector_database(index, index_path)

            # Save embeddings (optional, can be large)
            # embeddings_path = output_dir / 'embeddings.npy'
            # np.save(embeddings_path, embeddings)

            # Save metadata
            logger.info("Saving metadata...")
            metadata_path = output_dir / "metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump({"chunk_texts": chunk_texts, "sources": sources}, f)

            logger.info(
                f"Processing completed successfully. Outputs are saved in {output_dir}"
            )

        if query:
            if index_path is None:
                index_path = output_dir / "vector_database.index"
            if not index_path.exists():
                logger.error(
                    f"Index file not found at {index_path}. Please build the index or provide correct path."
                )
                sys.exit(1)
            index = load_vector_database(index_path)

            # Load metadata
            metadata_path = output_dir / "metadata.pkl"
            if not metadata_path.exists():
                logger.error(f"Metadata file not found at {metadata_path}.")
                sys.exit(1)
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            chunk_texts = metadata["chunk_texts"]
            sources = metadata["sources"]

            # Perform search
            results = search_vector_database(query, index, k=k)
            logger.info(f"Top {k} results for query: '{query}'")
            print("=" * 80)
            for idx, distance in results:
                idx = int(idx)
                chunk = chunk_texts[idx]
                source = sources[idx]
                print(f"Result Index: {idx}")
                print(f"Source File: {source}")
                print(f"Similarity Score (Lower is better): {distance:.4f}")
                print(f"Content Snippet:\n{chunk}\n")
                print("=" * 80)

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()





#    python script_name.py /path/to/file_or_directory --build_index --output_dir /path/to/output_directory


#    Example:

#    python script_name.py ./docs --build_index --output_dir ./output


# 2. **Querying the Index:**

#    To search for a query in the vector database, provide the `--query` parameter. Ensure that the index has been built and is available at the specified `--index_path` or within the `--output_dir`.

#    python script_name.py --query "Your search query here" --top_k 5 --output_dir /path/to/output_directory


#    Example:

#    python script_name.py --query "machine learning techniques" --top_k 3 --output_dir ./output


# 3. **Combined Operation:**

#    You can build the index and perform a query in a single command if desired.

#    python script_name.py ./docs --build_index --query "deep learning" --top_k 5 --output_dir ./output


# **Features Implemented:**

# - **PEP-8 Compliance:** The code strictly follows PEP-8 guidelines for readability and maintainability.
# - **Type Hints:** All functions include type hints using the `typing` module.
# - **Optimized Performance:** The tokenizer and model are loaded once and reused. Multiprocessing is used for parallel processing of files.
# - **Error Handling:** Comprehensive try-except blocks and logging for error tracking and graceful failures.
# - **Modularity:** Functions are well-structured and modular for ease of understanding and maintenance.
# - **Command-Line Interface:** Uses `argparse` for flexible and user-friendly command-line interactions.
# - **Scalability:** Designed to handle large datasets efficiently, making use of FAISS for scalable similarity search.

# **Explanation:**

# The script performs the following steps:

# 1. **Text Extraction:**
#    - Extracts text from PDF and TXT files using `PyMuPDF` and standard file I/O.
#    - Handles errors gracefully if files cannot be read.

# 2. **Text Chunking:**
#    - Splits the extracted text into manageable chunks based on the token count.
#    - Utilizes the tokenizer from the transformer model for accurate tokenization.

# 3. **Embedding Generation:**
#    - Generates embeddings for each text chunk using a pretrained transformer model.
#    - Mean pooling is applied to obtain a fixed-size vector representation.

# 4. **Vector Database Creation:**
#    - Builds a FAISS index (`IndexFlatL2`) for efficient similarity search.
#    - Stores the embeddings within the index for retrieval.

# 5. **Metadata Management:**
#    - Saves metadata including chunk texts and their source files to a pickle file.
#    - Ensures the order of embeddings and metadata align for accurate mapping during retrieval.

# 6. **Query Processing:**
#    - Accepts a user query and computes its embedding.
#    - Searches the FAISS index to find the top `k` most similar chunks.
#    - Retrieves and displays the corresponding text chunks and source information.

# 7. **Error Handling:**
#    - Checks for the existence of index and metadata files before querying.
#    - Provides meaningful error messages and exits gracefully on failure.

# **Notes:**

# - The script assumes that the embeddings correspond directly to the indices in the FAISS index and the metadata lists.
# - The similarity score returned by FAISS using `IndexFlatL2` is the L2 distance; lower distances indicate higher similarity.
# - For more accurate similarity measures, consider normalizing embeddings or using cosine similarity.

# **Dependencies:**

# - **PyMuPDF (`fitz`):** For PDF processing.
# - **Transformers (`transformers`):** For tokenizer and model.
# - **PyTorch (`torch`):** Required for transformer models.
# - **FAISS (`faiss`):** For efficient similarity search.
# - **NumPy (`numpy`):** For numerical operations.
# - **tqdm:** For progress bars during processing.
# - **Pickle:** For storing metadata.

# **Installation of Dependencies:**

# Make sure to install all required Python packages:

# pip install PyMuPDF transformers torch faiss-cpu numpy tqdm


# If you have a GPU and want to use it with FAISS:

# pip install faiss-gpu
