import json
import os
from typing import List, Dict, Optional,Any
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from g4f.client import Client

# Model Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': False}
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)

def load_json_files(folder_path: str) -> List[Dict[str, List[str]]]:
    """
    Load JSON files from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing JSON files.
    
    Returns:
        List[Dict[str, List[str]]]: List of dictionaries containing file contents.
    """
    json_data = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), 'r') as file:
                    json_data.extend(json.load(file))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON files: {e}")
    return json_data

def create_vector_store(data: List[Dict[str, List[str]]]) -> FAISS:
    """
    Create a FAISS vector store from the provided data.
    
    Args:
        data (List[Dict[str, List[str]]]): List of dictionaries containing file contents.
    
    Returns:
        FAISS: FAISS vector store.
    """
    documents = []
    for item in data:
        query = item.get('query', '')
        retrieve_content = ' '.join(item.get('retrieve_content', []))
        response = ' '.join(item.get('response', []))
        content = f"{query} {retrieve_content} {response}"
        documents.append(Document(page_content=content))
    
    return FAISS.from_documents(documents, EMBEDDING_MODEL, distance_strategy=DistanceStrategy.COSINE)

# def find_relevant_content(query: str, vector_store: FAISS, threshold: float = 0.99, max_results: int = 1) -> List[str]:
#     """
#     Find relevant content for the given query only if the most relevant result meets the threshold.
    
#     Args:
#         query (str): The query string.
#         vector_store (FAISS): FAISS vector store.
#         threshold (float): Similarity threshold (default: 0.99).
#         max_results (int): Maximum number of results to return (default: 1).
    
#     Returns:
#         List[str]: List of relevant content if the most relevant result meets the threshold, otherwise an empty list.
#     """
#     results = vector_store.similarity_search_with_score(query, k=max_results)
#     # Return content only if the most relevant result meets the threshold
#     if results and results[0][1] >= threshold:
#         return [result[0].page_content for result in results]
#     return []
def find_relevant_content(query: str, vector_store: FAISS, threshold: float = 0.99, max_results: int = 1) -> List[str]:
    """
    Find highly relevant content for the given query.
    
    Args:
        query (str): The query string.
        vector_store (FAISS): FAISS vector store.
        threshold (float): Similarity threshold (default: 0.99).
        max_results (int): Maximum number of results to return (default: 1).
    
    Returns:
        List[str]: List of highly relevant content or an empty list if no results meet the threshold.
    """
    try:
        results = vector_store.similarity_search_with_score(query, k=max_results)
        relevant_results = [result[0].page_content for result in results if result[1] >= threshold]
        return relevant_results if relevant_results else []
    except Exception as e:
        print(f"Error in finding relevant content: {e}")
        return []
def query_model(prompt:str,) -> Optional[str]:
    """
    Query the language model with the given query and context.
    
    Args:
        query (str): The query string.
        context (List[str]): List of relevant context.
    
    Returns:
        Optional[str]: Generated response or None if failed.
    """
    client = Client()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying model: {e}")
        return None




def load_history(filename: str = 'history.json') -> List[Dict[str, Any]]:
    """
    Load existing conversation history from a JSON file.
    
    Args:
        filename (str): Name of the file to load history from (default: 'history.json').
    
    Returns:
        List[Dict[str, Any]]: List of conversation history entries.
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {filename}. Starting with empty history.")
    return []

def save_history(history: List[Dict[str, Any]], filename: str = 'history.json'):
    """
    Append the conversation history to a JSON file.
    
    Args:
        history (List[Dict[str, Any]]): List of conversation history entries.
        filename (str): Name of the file to save history (default: 'history.json').
    """
    try:
        existing_history = load_history(filename)
        existing_history.extend(history)
        with open(filename, 'w') as f:
            json.dump(existing_history, f, indent=2)
        print(f"History appended to {filename}")
    except IOError as e:
        print(f"Error saving history: {e}")

def main():
    folder_path = "E:/LLMS/Fine-tuning/llms-data/history"
    data = load_json_files(folder_path)
    vector_store = create_vector_store(data)
    history = []

    print("Interactive query session started. Type 'exit' to quit.")
    while True:
        try:
            query = input("Enter your query:  and end the chat enter 'exit', 'quit' ").strip()
            if query.lower() in ['exit', 'quit']:
                break

            relevant_content = find_relevant_content(query, vector_store)
            prompt = (
                    f"Your task is to generate a well-crafted prompt for LLMs by understanding the given query and relevant content."
                    f"\n\nStep 1: Carefully interpret the Query '{query}' and the relevant content '{relevant_content}'."
                    f"\nStep 2: Identify the key points and objectives within the query and content."
                    f"\nStep 3: If the relevant content is insufficient or missing, reformulate the query to be as comprehensive as possible."
                    f"\nStep 4: Rewrite the information to be optimized for LLM understanding and processing."
                    f"\nStep 5: Ensure the output is tailored specifically for LLM input, maintaining clarity and precision."
                )


            response_summary = query_model(prompt)
            prompt_final = (
                            f"Please analyze the problem step-by-step, using your complete knowledge and skills to arrive at a comprehensive response.\n\n"
                            f"TASK: {response_summary}\n\n"
                            f"Step 1: Understand the core components of the problem.\n"
                            f"Step 2: Identify and describe potential approaches to the problem.\n"
                            f"Step 3: Evaluate the pros and cons of each approach.\n"
                            f"Step 4: Synthesize the findings into a coherent and actionable response."
          )

            response = query_model(prompt=prompt_final)

            if response:
                history_entry = {
                    "query": query,
                    "retrieved_content": relevant_content,
                    "response": response
                }
                history.append(history_entry)
                print("\n \n  🧠 🧠 🧠  Response:", response)
            else:
                print("Failed to generate a response. Please try again.")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

    save_history(history)

if __name__ == "__main__":
    main()



# import os
# import json
# from typing import List, Dict, Tuple, Union
# from tqdm import tqdm
# from langchain_community.vectorstores.faiss import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# from g4f.client import Client
# from langchain_community.vectorstores.utils import DistanceStrategy


# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# MODEL_KWARGS = {'device': 'cpu'}
# ENCODE_KWARGS = {'normalize_embeddings': False}
# EMBEDDING_MODEL = HuggingFaceEmbeddings(
#     model_name=MODEL_NAME,
#     model_kwargs=MODEL_KWARGS,
#     encode_kwargs=ENCODE_KWARGS
# )


# def load_json_files(directory: str) -> List[Dict[str, Union[str, List[str]]]]:
#     """
#     Load JSON files from a given directory.
#     """
#     json_data = []
#     for filename in os.listdir(directory):
#         path = os.path.join(directory, filename)
#         if path.endswith('.json'):
#             with open(path, 'r') as file:
#                 json_data.append(json.load(file))
#     return json_data


# def find_relevant_content(query: str, retrieved_content: List[str]) -> List[str]:
#     """
#     Find relevant content based on the query from the retrieved content.
#     Returns content with 95% or higher relevance.
#     """
#     documents = [Document(page_content=text) for text in retrieved_content]
#     vector_store = FAISS.from_documents(
#         documents, EMBEDDING_MODEL.embeddings, distance_strategy=DistanceStrategy.COSINE
#     )

#     try:
#         results = vector_store.similarity_search_with_score(query, threshold=0.95)
#         relevant_content = [result[0].page_content for result in results]
#     except Exception as e:
#         print(f"Error during similarity search: {e}")
#         relevant_content = []
    
#     return relevant_content


# def query_model(query: str, context: List[str]) -> str:
#     """
#     Query the model with the given query and context.
#     """
#     client = Client()
#     retrieve_content = " ".join(context)
#     prompt = f"Query: {query} {retrieve_content} given response in great detail novelity and think about query and generates properly regarding query"

#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"Error during querying the model: {e}")
#         return ""


# def save_history(history: List[Dict[str, Union[str, List[str]]]], filename: str = 'query_history.json') -> None:
#     """
#     Save history to a JSON file.
#     """
#     with open(filename, 'w') as file:
#         json.dump(history, file, indent=2)


# def main() -> None:
#     history = []

#     while True:
#         try:
#             query = input("Enter your query (or type 'exit' to quit): ").strip()
#             if query.lower() in ['exit', 'quit']:
#                 break

#             json_data = load_json_files('your_directory_path_here')
#             retrieved_content = []

#             for data in json_data:
#                 retrieved_content.extend(data.get('retrieve_content', []))
#                 previous_responses = data.get('response', [])
#                 retrieved_content.extend(previous_responses)

#             context = find_relevant_content(query=query, retrieved_content=retrieved_content)
#             response = query_model(query, context=context)

#             if response:
#                 history_entry = {
#                     "query": query,
#                     "retrieved_content": context,
#                     "response": response
#                 }
#                 history.append(history_entry)
#                 print(response)
#             else:
#                 print("Failed to generate a response. Please try again.")

#         except KeyboardInterrupt:
#             print("\nSession interrupted. Exiting.")
#             break

#     save_history(history)


# if __name__ == '__main__':
#     main()