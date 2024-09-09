import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import faiss
from pathlib import Path
from transformers import AutoTokenizer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from datacollection import AdvancedDirectoryLoader
from document_splitter import AdvancedDocumentSplitter
from embedding_data import AdvancedFAISS
import yaml
import pacmap
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import emoji

# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def process_document(doc: str, splitter: AdvancedDocumentSplitter) -> List[Dict[str, Any]]:
    try:
        return splitter.split_documents([doc])
    except Exception as e:
        print(f"Error processing document: {e}")
        return []

def build_vector_database(
    data_dir: str, 
    embedding_model_name: str, 
    chunk_size: int
) -> Tuple[Optional[AdvancedFAISS], Optional[HuggingFaceEmbeddings], List[Dict[str, Any]]]:
    try:
        loader = AdvancedDirectoryLoader(data_dir, exclude=[".pyc", "__pycache__"], silent_errors=True)
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        splitter = AdvancedDocumentSplitter(tokenizer=tokenizer, chunk_size=chunk_size)

        documents = loader.load()

        # Process documents sequentially
        docs_processed = [
            processed_doc
            for doc in documents
            for processed_doc in process_document(doc, splitter)
        ]

        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        sample_embedding = embeddings_model.embed_query("Sample text")
        dimension = len(sample_embedding)
        index = faiss.IndexFlatL2(dimension)
        advanced_faiss = AdvancedFAISS(
            embedding_function=embeddings_model.embed_query,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        faiss_index = advanced_faiss.from_documents(
            docs_processed, embeddings_model, distance_strategy=DistanceStrategy.COSINE
        )

        return faiss_index, embeddings_model, docs_processed

    except Exception as e:
        print(f"Error building vector database: {e}")
        return None, None, []

def format_prompt(retrieved_docs: List[Dict[str, Any]], question: str) -> str:
    try:
        retrieved_docs_text = [doc['page_content'] for doc in retrieved_docs]
        context = "\nExtracted documents:\n" + "".join(
            [f"Document {str(i)}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)]
        )

        prompt_template = [
            {
                "role": "system",
                "content": """Using the information contained in the context, 
give a comprehensive answer to the question. 
Respond only to the question asked; response should be concise and relevant to the question. 
Provide the number of the source document when relevant. 
If the answer cannot be deduced from the context, try to give the best of your ability."""
            },
            {
                "role": "user",
                "content": f"""Context:
{context}
---
Now here is the question you need to answer.

Question: {question}"""
            }
        ]

        return json.dumps(prompt_template, indent=2)

    except Exception as e:
        print(f"Error formatting prompt: {e}")
        return ""

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
    except Exception as e:
        print(f"Unexpected error loading configuration file: {e}")
    return {}



def get_emoji(text: str) -> str:
    """Get a relevant emoji based on the text content."""
    text_lower = text.lower()
    if 'code' in text_lower:
        return emoji.emojize(':computer:', language='alias')
    elif 'data' in text_lower:
        return emoji.emojize(':bar_chart:', language='alias')
    elif 'question' in text_lower:
        return emoji.emojize(':question:', language='alias')
    elif 'file' in text_lower or 'document' in text_lower:
        return emoji.emojize(':file_folder:', language='alias')
    elif 'image' in text_lower or 'picture' in text_lower:
        return emoji.emojize(':frame_with_picture:', language='alias')
    elif 'text' in text_lower or 'content' in text_lower:
        return emoji.emojize(':memo:', language='alias')
    else:
        return emoji.emojize(':page_facing_up:', language='alias')
    
def get_symbol(text: str) -> str:
    """Get a relevant symbol based on the text content."""
    text_lower = text.lower()
    if 'code' in text_lower:
        return '💻'
    elif 'data' in text_lower:
        return '📊'
    elif 'question' in text_lower:
        return '❓'
    elif 'file' in text_lower or 'document' in text_lower:
        return '📁'
    elif 'image' in text_lower or 'picture' in text_lower:
        return '🖼️'
    elif 'text' in text_lower or 'content' in text_lower:
        return '📝'
    else:
        return '📄'

def create_advanced_visualization(df: pd.DataFrame, user_query: str) -> go.Figure:
    """Create an advanced, interactive visualization of document embeddings."""
    df['symbol'] = df['extract'].apply(get_symbol)
    df['emoji'] = df['extract'].apply(get_emoji)
    df['size_col'] = np.where(df['source'] == 'User query', 30, 15)
    
    unique_sources = df['source'].unique()
    color_map = {source: f'hsl({i*360/len(unique_sources)},70%,50%)' for i, source in enumerate(unique_sources)}
    df['color'] = df['source'].map(color_map)
    
    fig = go.Figure()
    
    for source in unique_sources:
        source_df = df[df['source'] == source]
        fig.add_trace(go.Scatter(
            x=source_df['x'],
            y=source_df['y'],
            mode='markers+text',
            marker=dict(
                size=source_df['size_col'],
                color=source_df['color'],
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=source_df['symbol'],  # Use symbol for visibility
            textposition="middle center",
            hoverinfo='text',
            hovertext=source_df.apply(lambda row: (
                f"Source: {row['source']}<br>"
                f"Extract: {row['extract']}<br>"
                f"Emoji: {row['emoji']}"
            ), axis=1),
            name=source
        ))
    
    fig.update_layout(
        title={
            'text': "Document Embedding Visualization",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        showlegend=True,
        legend_title="Document Sources",
        hovermode='closest',
        template='plotly_dark',
        height=800,  # Increase height for better visibility
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Add annotation for user query
    user_query_point = df[df['source'] == 'User query']
    if not user_query_point.empty:
        fig.add_annotation(
            x=user_query_point['x'].values[0],
            y=user_query_point['y'].values[0],
            text="User Query",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-40,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
        )
    
    # Add info table
    fig.add_trace(
        go.Table(
            header=dict(values=["Query Information"],
                        fill_color='rgba(255, 140, 0, 0.8)',
                        align='left'),
            cells=dict(values=[[f"Query: {user_query}",
                                f"Total Documents: {len(df) - 1}",
                                f"Unique Sources: {len(unique_sources) - 1}"]],
                       fill_color='rgba(100, 100, 100, 0.8)',
                       align='left'),
            domain=dict(x=[0, 0.3], y=[0, 0.2])
        )
    )
    
    return fig
def main(file_path):
    config = load_config(config_path=file_path)
    if not config:
        print("Exiting due to configuration load failure.")
        return

    try:
        rag_config = config["CONFIG"]
        user_query = rag_config["QUERY"]

        vector_db, embeddings_model, docs_processed = build_vector_database(
            data_dir=rag_config["DATA_DIR"],
            embedding_model_name=rag_config["EMBEDDING_MODEL_NAME"],
            chunk_size=rag_config["CHUNK_SIZE"],
        )
        
        if vector_db:
            retrieved_docs = vector_db.similarity_search(user_query, k=rag_config["NUM_SIMILAR"])
            final_prompt = format_prompt(retrieved_docs, user_query)

            query_vector = embeddings_model.embed_query(user_query)

            # Visualization
            embedding_projector = pacmap.PaCMAP(
                n_components=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
            )

            embeddings_2d = [
                list(vector_db.index.reconstruct_n(idx, 1)[0])
                for idx in range(len(docs_processed))
            ] + [query_vector]

            documents_projected = embedding_projector.fit_transform(
                np.array(embeddings_2d), init="pca"
            )

            df = pd.DataFrame([
                {
                    "x": documents_projected[i, 0],
                    "y": documents_projected[i, 1],
                    "z": documents_projected[i, 2],
                    "source": docs_processed[i].metadata.get("file_path", "Unknown").split("/")[-1],
                    "extract": docs_processed[i].page_content[:150] + "...",
                    "type": "Document"
                }
                for i in range(len(docs_processed))
            ] + [
                {
                    "x": documents_projected[-1, 0],
                    "y": documents_projected[-1, 1],
                    "z": documents_projected[-1, 2],
                    "source": "User query",
                    "extract": user_query,
                    "type": "Query"
                }
            ])
            fig=create_advanced_visualization(df=df ,user_query=user_query)
            fig.write_html("Query_embedding_visualization.html")
            print("Visualization saved as novel_embedding_visualization.html")
        else:
            print("Failed to build vector database.")

    except KeyError as e:
        print(f"Missing configuration key: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
