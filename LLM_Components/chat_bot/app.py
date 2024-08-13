import os
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
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
        self.data_dir = data_dir
        self.model_name = model_name
        self.loader = None
        self.splitter = None
        self.embeddings_model = None
        self.advanced_faiss = None
        self.reader = None
        self.stt_pipeline = None
        self.tts = None

    def initialize_components(self):
        self._initialize_loader()
        self._initialize_splitter()
        self._initialize_embeddings()
        self._initialize_faiss()
        self._initialize_reader()
        self._initialize_speech_components()

    def _initialize_loader(self):
        self.loader = AdvancedDirectoryLoader(self.data_dir)

    def _initialize_splitter(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.splitter = AdvancedDocumentSplitter(
            tokenizer=tokenizer,
            chunk_size=500
        )

    def _initialize_embeddings(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _initialize_faiss(self):
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

    def _initialize_reader(self):
        self.reader = ReadPipeline(
            model_name=self.model_name,
            quantization="map"
        )

    def _initialize_speech_components(self):
        self.stt_pipeline = SpeechToTextPipeline("openai/whisper-small")
        self.tts = TextToSpeechAPI()

    def load_and_process_documents(self) -> List[Dict[str, Any]]:
        documents = self.loader.load()
        split_documents = self.splitter.split_documents(documents)
        texts = [doc.page_content for doc in split_documents]
        metadatas = [doc.metadata for doc in split_documents]
        self.advanced_faiss.add_texts(texts=texts, metadatas=metadatas)
        return split_documents

    def perform_similarity_search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        return self.advanced_faiss.similarity_search(query, k=k)

    def perform_mmr_search(self, query: str, k: int = 2, fetch_k: int = 3, lambda_mult: float = 0.5) -> List[Dict[str, Any]]:
        return self.advanced_faiss.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

    def generate_response(self, context: str, question: str) -> str:
        return self.reader.generate_response(context=context, question=question)

    def transcribe_audio_file(self, audio_file: str) -> str:
        audio = ffmpeg_read(open(audio_file, "rb").read(), self.stt_pipeline.sampling_rate)
        result = self.stt_pipeline.transcribe(audio)
        return result['text']

    def transcribe_microphone(self, duration: int) -> str:
        microphone = ffmpeg_microphone(self.stt_pipeline.sampling_rate, duration)
        audio = np.frombuffer(next(microphone), dtype=np.float32)
        result = self.stt_pipeline.transcribe(audio)
        return result['text']

    def transcribe_stream(self, duration: int, chunk_length_s: float = 0.5) -> List[str]:
        transcriptions = []
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

        return transcriptions

    def text_to_speech(self, text: str, output_file: str = None, rate: int = 200, volume: float = 1.0) -> None:
        if output_file:
            self.tts.save_to_file(text, output_file, rate=rate)
        else:
            self.tts.speak(text, rate=rate, volume=volume)

    def run(self):
        self.initialize_components()
        documents = self.load_and_process_documents()
        
        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            search_results = self.perform_similarity_search(query)
            context = " ".join([doc.page_content for doc in search_results])
            response = self.generate_response(context, query)
            
            print(f"Response: {response}")
            self.text_to_speech(response)

            # Example of speech-to-text usage
            print("Speak your next query (5 seconds):")
            spoken_query = self.transcribe_microphone(5)
            print(f"Transcribed query: {spoken_query}")

if __name__ == "__main__":
    app = AdvancedRAGApp(data_dir="path/to/your/documents", model_name="GPT-4o")
    app.run()