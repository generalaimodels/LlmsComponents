import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import gradio as gr
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
import faiss
from datacollection import AdvancedDirectoryLoader
from document_splitter import AdvancedDocumentSplitter
from embedding_data import AdvancedFAISS
from text_generation import ReadPipeline
from speechtotext import (
    ffmpeg_read,
    ffmpeg_microphone,
    ffmpeg_microphone_live,
    SpeechToTextPipeline
)
from text_to_speech import TextToSpeechAPI


class AdvancedRAGApp:
    def __init__(self, data_dir: str, model_name: str):
        self.data_dir: str = data_dir
        self.model_name: str = model_name
        self.loader: Optional[AdvancedDirectoryLoader] = None
        self.splitter: Optional[AdvancedDocumentSplitter] = None
        self.embeddings_model: Optional[HuggingFaceEmbeddings] = None
        self.advanced_faiss: Optional[AdvancedFAISS] = None
        self.reader: Optional[ReadPipeline] = None
        self.stt_pipeline: Optional[SpeechToTextPipeline] = None
        self.tts: Optional[TextToSpeechAPI] = None
        self.history: List[Dict[str, List[str]]] = []

    def initialize_components(self) -> None:
        try:
            self._initialize_loader()
            self._initialize_splitter()
            self._initialize_embeddings()
            self._initialize_faiss()
            self._initialize_reader()
            self._initialize_speech_components()
        except Exception as e:
            print(f"Error initializing components: {e}")

    def _initialize_loader(self) -> None:
        self.loader = AdvancedDirectoryLoader(self.data_dir)

    def _initialize_splitter(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.splitter = AdvancedDocumentSplitter(
            tokenizer=tokenizer,
            chunk_size=500
        )

    def _initialize_embeddings(self) -> None:
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _initialize_faiss(self) -> None:
        sample_embedding = self.embeddings_model.embed_query("Sample text")
        dimension = len(sample_embedding)
        index = faiss.IndexFlatL2(dimension)
        self.advanced_faiss = AdvancedFAISS(
            embedding_function=self.embeddings_model.embed_query,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
        )

    def _initialize_reader(self) -> None:
        self.reader = ReadPipeline(
            model_name=self.model_name,
            quantization="map"
        )

    def _initialize_speech_components(self) -> None:
        self.stt_pipeline = SpeechToTextPipeline("openai/whisper-small")
        self.tts = TextToSpeechAPI()

    def load_and_process_documents(self) -> List[Dict[str, Any]]:
        try:
            documents = self.loader.load()
            split_documents = self.splitter.split_documents(documents)
            texts = [doc.page_content for doc in split_documents]
            metadatas = [doc.metadata for doc in split_documents]
            self.advanced_faiss.add_texts(texts=texts, metadatas=metadatas)
            return split_documents
        except Exception as e:
            print(f"Error loading and processing documents: {e}")
            return []

    def perform_similarity_search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        try:
            return self.advanced_faiss.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []

    def perform_mmr_search(self, query: str, k: int = 2, fetch_k: int = 3, lambda_mult: float = 0.5) -> List[Dict[str, Any]]:
        try:
            return self.advanced_faiss.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        except Exception as e:
            print(f"Error performing MMR search: {e}")
            return []

    def generate_response(self, context: str, question: str) -> str:
        try:
            return self.reader.generate_response(context=context, question=question)
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def transcribe_audio_file(self, audio_file: str) -> str:
        try:
            with open(audio_file, "rb") as file:
                audio = ffmpeg_read(file.read(), self.stt_pipeline.sampling_rate)
            result = self.stt_pipeline.transcribe(audio)
            return result['text']
        except Exception as e:
            print(f"Error transcribing audio file: {e}")
            return ""

    def transcribe_microphone(self, duration: int) -> str:
        try:
            microphone = ffmpeg_microphone(self.stt_pipeline.sampling_rate, duration)
            audio = np.frombuffer(next(microphone), dtype=np.float32)
            result = self.stt_pipeline.transcribe(audio)
            return result['text']
        except Exception as e:
            print(f"Error transcribing microphone: {e}")
            return ""

    def transcribe_stream(self, duration: int, chunk_length_s: float = 0.5) -> List[str]:
        transcriptions = []
        try:
            stream = ffmpeg_microphone_live(
                sampling_rate=self.stt_pipeline.sampling_rate,
                chunk_length_s=chunk_length_s,
                stream_chunk_s=1,
            )

            for i, chunk in enumerate(stream):
                if i * chunk_length_s >= duration:
                    break

                if not chunk["partial"]:
                    result = self.stt_pipeline.transcribe(chunk["raw"])
                    transcriptions.append(result['text'])
        except Exception as e:
            print(f"Error transcribing stream: {e}")

        return transcriptions

    def text_to_speech(self, text: str, output_file: Optional[str] = None, rate: int = 200, volume: float = 1.0) -> None:
        try:
            if output_file:
                self.tts.save_to_file(text, output_file, rate=rate)
            else:
                self.tts.speak(text, rate=rate, volume=volume)
        except Exception as e:
            print(f"Error in text to speech: {e}")

    def update_history(self, query: str, response: str) -> None:
        try:
            self.history.append({"query": [query], "response": [response]})
        except Exception as e:
            print(f"Error updating history: {e}")

    def get_history(self) -> List[Dict[str, List[str]]]:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()

    def save_history(self, filename: str) -> None:
        try:
            with open(filename, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Error saving history: {e}")

    def load_history(self, filename: str) -> None:
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")


def gradio_interface(
    data_dir: str,
    model_name: str,
    query: str
) -> Tuple[str, List[Dict[str, List[str]]]]:
    """Function to integrate with Gradio for a web-based interface."""
    try:
        app = AdvancedRAGApp(data_dir, model_name)
        app.initialize_components()
        app.load_and_process_documents()

        search_results = app.perform_similarity_search(query)
        context = " ".join([doc.page_content for doc in search_results])
        response = app.generate_response(context, query)
        app.update_history(query, response)

        return response, app.get_history()

    except Exception as e:
        return str(e), []


gr_interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Data Directory", lines=1),
        gr.Textbox(label="Model Name", lines=1),
        gr.Textbox(label="Enter Query", lines=2)
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.JSON(label="Interaction History")
    ],
    title="Advanced RAG App",
    description="An interactive Gradio interface for the AdvancedRAGApp."
)

if __name__ == "__main__":
    gr_interface.launch(share=True)