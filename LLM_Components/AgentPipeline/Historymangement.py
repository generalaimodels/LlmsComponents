import json
import os
from typing import List, Dict, Optional
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

def process_folder(folder_path: str, distance_strategy: DistanceStrategy) -> None:
    """
    Process JSON files in the given folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        distance_strategy (DistanceStrategy): Distance strategy for similarity search.

    Raises:
        FileNotFoundError: If the folder path is invalid.
        json.JSONDecodeError: If there's an error decoding JSON.
    """
    try:
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                process_file(file_path, distance_strategy)
    except FileNotFoundError:
        print(f"Error: Folder not found at {folder_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def process_file(file_path: str, distance_strategy: DistanceStrategy) -> None:
    """
    Process a single JSON file.

    Args:
        file_path (str): Path to the JSON file.
        distance_strategy (DistanceStrategy): Distance strategy for similarity search.

    Raises:
        json.JSONDecodeError: If there's an error decoding JSON.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        for item in data:
            query = item.get('query', '')
            retrieve_content = item.get('retrieve_content', [])
            previous_response = item.get('response', [])

            if query:
                retrieved_content = search_relevant_content(query, retrieve_content, previous_response, distance_strategy)
                response = generate_response(query, retrieved_content)
                
                item['retrieved_content'] = retrieved_content
                item['response'] = response

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
    except Exception as e:
        print(f"An error occurred while processing file {file_path}: {str(e)}")

def search_relevant_content(query: str, retrieve_content: List[str], previous_response: List[str], distance_strategy: DistanceStrategy) -> List[str]:
    """
    Search for relevant content based on the query.

    Args:
        query (str): The search query.
        retrieve_content (List[str]): List of content to search from.
        previous_response (List[str]): List of previous responses.
        distance_strategy (DistanceStrategy): Distance strategy for similarity search.

    Returns:
        List[str]: List of relevant content.
    """
    documents = [Document(page_content=text) for text in retrieve_content + previous_response]
    
    try:
        vectorstore = FAISS.from_documents(documents, EMBEDDING_MODEL, distance_strategy=distance_strategy)
        results = vectorstore.similarity_search_with_relevance_scores(query, k=5)
        
        relevant_content = [doc.page_content for doc, score in results if score >= 0.95]
        return relevant_content
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
        return []

def generate_response(query: str, retrieved_content: List[str]) -> str:
    """
    Generate a response using the G4F client.

    Args:
        query (str): The original query.
        retrieved_content (List[str]): List of retrieved relevant content.

    Returns:
        str: Generated response.
    """
    client = Client()
    prompt = f"Query: {query}\nRetrieved Content: {retrieved_content}\nPlease provide a detailed and novel response regarding the query."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return ""

# if __name__ == "__main__":
#     folder_path = input("Enter the folder path containing JSON files: ")
#     distance_strategy = DistanceStrategy.COSINE  # You can change this to other strategies if needed
#     process_folder(folder_path, distance_strategy)
    
    


import json
import os
from typing import List, Dict
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

def load_json_files(directory: str) -> List[Dict]:
    """Load all JSON files from a specified directory."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        json_files.append(data)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error loading {file_path}: {e}")
    return json_files

def search_relevant_content(query: str, retrieve_content: List[str]) -> List[str]:
    """Search and return content with at least 95% relevance."""
    documents = [Document(page_content=text) for text in retrieve_content]
    vectorstore = FAISS.from_documents(documents, EMBEDDING_MODEL)

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query, distance_strategy=DistanceStrategy.COSINE
        )
        return [doc.page_content for doc, score in results if score >= 0.95]
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []

def generate_response(query: str, relevant_content: List[str]) -> str:
    """Generate a detailed response using a language model."""
    client = Client()
    content_str = " ".join(relevant_content)
    prompt = f"Query: {query} {content_str} given response in great details novelty and think about query and generates properly regarding query"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred generating the response."

def process_json_files(directory: str) -> None:
    """Process all JSON files in a directory, updating them with retrieved content and responses."""
    json_files = load_json_files(directory)

    for file_data in tqdm(json_files, desc="Processing JSON files"):
        query = file_data.get('query')
        retrieve_content = file_data.get('retrieve_content', [])
        
        if query:
            relevant_content = search_relevant_content(query, retrieve_content)
            response = generate_response(query, relevant_content)
            file_data['retrieved_content'] = relevant_content
            file_data['response'] = response

            # Save the updated JSON data back to the file
            try:
                with open(file_data['file_path'], 'w', encoding='utf-8') as f:
                    json.dump(file_data, f, indent=4)
            except FileNotFoundError as e:
                print(f"Error saving file: {e}")

# def main():
#     directory = "path/to/your/json/folder"
#     process_json_files(directory)

# if __name__ == "__main__":
#     main()

import json
import os
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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

def load_json_files(directory: str) -> List[Dict]:
    """Load all JSON files from a specified directory."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['file_path'] = file_path  # Add file path to data for saving later
                        json_files.append(data)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error loading {file_path}: {e}")
    return json_files

def search_relevant_content(query: str, retrieve_content: List[str]) -> List[str]:
    """Search and return content with at least 95% relevance using an adaptive threshold."""
    documents = [Document(page_content=text) for text in retrieve_content]
    vectorstore = FAISS.from_documents(documents, EMBEDDING_MODEL)

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query, distance_strategy=DistanceStrategy.COSINE
        )
        threshold = compute_adaptive_threshold(results)
        relevant_content = [doc.page_content for doc, score in results if score >= threshold]
        return relevant_content
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []

def compute_adaptive_threshold(results):
    """Compute an adaptive threshold based on score distribution."""
    scores = [score for _, score in results]
    if not scores:
        return 0.95
    mean_score = sum(scores) / len(scores)
    adjusted_threshold = max(0.95, mean_score * 0.95)
    return adjusted_threshold

def generate_response(query: str, relevant_content: List[str]) -> str:
    """Generate a detailed response using a language model."""
    client = Client()
    content_str = " ".join(relevant_content)
    prompt = (
        f"Query: {query} "
        f"{content_str} "
        "given response in great detail, novelty and think about query and generate properly regarding the query"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred generating the response."

def process_file(file_data: Dict) -> None:
    """Process a single JSON file, updating with retrieved content and responses."""
    query = file_data.get('query')
    retrieve_content = file_data.get('retrieve_content', [])
    
    if query:
        relevant_content = search_relevant_content(query, retrieve_content)
        response = generate_response(query, relevant_content)
        file_data['retrieved_content'] = relevant_content
        file_data['response'] = response

        # Save the updated JSON data back to the file
        try:
            with open(file_data['file_path'], 'w', encoding='utf-8') as f:
                json.dump(file_data, f, indent=4)
        except FileNotFoundError as e:
            print(f"Error saving file: {e}")

def process_json_files(directory: str) -> None:
    """Process all JSON files in a directory in parallel."""
    json_files = load_json_files(directory)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, json_files), total=len(json_files), desc="Processing JSON files"))




import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from g4f.client import Client
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Model Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': False}
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)

def process_folder(folder_path: str, distance_strategy: DistanceStrategy) -> None:
    try:
        with ThreadPoolExecutor() as executor:
            futures = []
            for filename in os.listdir(folder_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(folder_path, filename)
                    futures.append(executor.submit(process_file, file_path, distance_strategy))
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
    except FileNotFoundError:
        print(f"Error: Folder not found at {folder_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def process_file(file_path: str, distance_strategy: DistanceStrategy) -> None:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        updated_data = []
        for item in data:
            query = item.get('query', '')
            retrieve_content = item.get('retrieve_content', [])
            previous_response = item.get('response', [])

            if query:
                retrieved_content = hybrid_search(query, retrieve_content, previous_response, distance_strategy)
                response = generate_dynamic_response(query, retrieved_content)
                
                updated_item = {
                    'query': query,
                    'retrieved_content': retrieved_content,
                    'response': response,
                    'metadata': generate_metadata(query, retrieved_content, response)
                }
                updated_data.append(updated_item)

        with open(file_path, 'w') as file:
            json.dump(updated_data, file, indent=2)

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
    except Exception as e:
        print(f"An error occurred while processing file {file_path}: {str(e)}")

def hybrid_search(query: str, retrieve_content: List[str], previous_response: List[str], distance_strategy: DistanceStrategy) -> List[str]:
    documents = [Document(page_content=text) for text in retrieve_content + previous_response]
    
    try:
        # Semantic Search
        vectorstore = FAISS.from_documents(documents, EMBEDDING_MODEL, distance_strategy=distance_strategy)
        semantic_results = vectorstore.similarity_search_with_relevance_scores(query, k=10)
        
        # Keyword Search
        tfidf = TfidfVectorizer().fit_transform([query] + retrieve_content + previous_response)
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        keyword_results = sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)[:10]
        
        # Combine and rank results
        combined_results = {}
        for doc, score in semantic_results:
            combined_results[doc.page_content] = score
        
        for idx, score in keyword_results:
            content = retrieve_content[idx] if idx < len(retrieve_content) else previous_response[idx - len(retrieve_content)]
            if content in combined_results:
                combined_results[content] += score
            else:
                combined_results[content] = score
        
        ranked_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        return [content for content, _ in ranked_results if _ >= 0.8]  # Adjusted threshold
    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        return []

def generate_dynamic_response(query: str, retrieved_content: List[str]) -> str:
    client = Client()
    
    # Dynamic prompt generation based on content analysis
    content_summary = summarize_content(retrieved_content)
    sentiment = analyze_sentiment(query)
    
    prompt = f"""Query: {query}
Retrieved Content Summary: {content_summary}
Detected Sentiment: {sentiment}

Please provide a detailed, insightful, and novel response to the query. 
Consider the following aspects:
1. Address the main points of the query
2. Incorporate relevant information from the retrieved content
3. Match the tone of the response to the detected sentiment
4. Provide unique perspectives or insights not directly stated in the retrieved content
5. If appropriate, suggest potential follow-up questions or areas for further exploration

Your response should be comprehensive yet concise, aiming to provide maximum value to the user."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return ""

def summarize_content(content: List[str]) -> str:
    # Implement a simple extractive summarization
    sentences = [sent for text in content for sent in text.split('.')]
    tfidf = TfidfVectorizer().fit_transform(sentences)
    sentence_scores = tfidf.sum(axis=1).A1
    top_sentences = sorted(enumerate(sentence_scores), key=lambda x: x[1], reverse=True)[:3]
    summary = ' '.join([sentences[idx] for idx, _ in top_sentences])
    return summary

def analyze_sentiment(text: str) -> str:
    # Simple rule-based sentiment analysis
    positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'])
    negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing'])
    
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

def generate_metadata(query: str, retrieved_content: List[str], response: str) -> Dict[str, Any]:
    return {
        "query_length": len(query),
        "content_count": len(retrieved_content),
        "response_length": len(response),
        "timestamp": datetime.now().isoformat(),
        "content_topics": extract_topics(retrieved_content)
    }




def extract_topics(content: List[str]) -> List[str]:
    # Simple topic extraction using TF-IDF and N-grams
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(content)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top 5 terms as topics
    top_topics = []
    for row in tfidf_matrix.toarray():
        top_indices = row.argsort()[-5:][::-1]
        top_topics.extend([feature_names[i] for i in top_indices])
    
    return list(set(top_topics))  # Remove duplicates

def calculate_content_diversity(retrieved_content: List[str]) -> float:
    # Calculate content diversity using Jaccard similarity
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    tokenized_contents = [set(word_tokenize(content.lower())) for content in retrieved_content]
    similarities = []
    for i in range(len(tokenized_contents)):
        for j in range(i+1, len(tokenized_contents)):
            similarities.append(jaccard_similarity(tokenized_contents[i], tokenized_contents[j]))
    
    return 1 - (sum(similarities) / len(similarities)) if similarities else 0

def generate_metadata(query: str, retrieved_content: List[str], response: str) -> Dict[str, Any]:
    return {
        "query_length": len(query),
        "content_count": len(retrieved_content),
        "response_length": len(response),
        "timestamp": datetime.now().isoformat(),
        "content_topics": extract_topics(retrieved_content),
        "content_diversity": calculate_content_diversity(retrieved_content),
        "query_complexity": calculate_query_complexity(query)
    }

def calculate_query_complexity(query: str) -> float:
    # Calculate query complexity based on unique words and sentence structure
    words = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    unique_words = set(word for word in words if word not in stop_words)
    
    word_complexity = len(unique_words) / len(words)
    sentence_complexity = len(query) / (query.count('.') + query.count('!') + query.count('?') + 1)
    
    return (word_complexity + sentence_complexity) / 2

def adaptive_threshold(query: str, retrieved_content: List[str]) -> float:
    # Dynamically adjust the relevance threshold based on query and content characteristics
    query_length = len(query.split())
    content_lengths = [len(content.split()) for content in retrieved_content]
    avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    
    base_threshold = 0.8
    length_factor = min(query_length / avg_content_length, 1) if avg_content_length > 0 else 1
    diversity_factor = calculate_content_diversity(retrieved_content)
    
    return base_threshold * (1 + length_factor) * (1 + diversity_factor) / 3

def hybrid_search(query: str, retrieve_content: List[str], previous_response: List[str], distance_strategy: DistanceStrategy) -> List[str]:
    documents = [Document(page_content=text) for text in retrieve_content + previous_response]
    
    try:
        # Semantic Search
        vectorstore = FAISS.from_documents(documents, EMBEDDING_MODEL, distance_strategy=distance_strategy)
        semantic_results = vectorstore.similarity_search_with_relevance_scores(query, k=10)
        
        # Keyword Search
        tfidf = TfidfVectorizer().fit_transform([query] + retrieve_content + previous_response)
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        keyword_results = sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)[:10]
        
        # Combine and rank results
        combined_results = {}
        for doc, score in semantic_results:
            combined_results[doc.page_content] = score
        
        for idx, score in keyword_results:
            content = retrieve_content[idx] if idx < len(retrieve_content) else previous_response[idx - len(retrieve_content)]
            if content in combined_results:
                combined_results[content] += score
            else:
                combined_results[content] = score
        
        ranked_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        
        # Use adaptive threshold
        threshold = adaptive_threshold(query, [content for content, _ in ranked_results])
        return [content for content, score in ranked_results if score >= threshold]
    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        return []

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing JSON files: ")
    distance_strategy = DistanceStrategy.COSINE
    process_folder(folder_path, distance_strategy)