import os
from typing import List, Dict, Any, Optional
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
        self._initialize_loader()
        self._initialize_splitter()
        self._initialize_embeddings()
        self._initialize_faiss()
        self._initialize_reader()
        self._initialize_speech_components()

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
        with open(audio_file, "rb") as file:
            audio = ffmpeg_read(file.read(), self.stt_pipeline.sampling_rate)
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

    def text_to_speech(self, text: str, output_file: Optional[str] = None, rate: int = 200, volume: float = 1.0) -> None:
        if output_file:
            self.tts.save_to_file(text, output_file, rate=rate)
        else:
            self.tts.speak(text, rate=rate, volume=volume)

    def update_history(self, query: str, response: str) -> None:
        self.history.append({"query": [query], "response": [response]})

    def run(self) -> None:
        self.initialize_components()
        self.load_and_process_documents()
        
        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            search_results = self.perform_similarity_search(query)
            context = " ".join([doc.page_content for doc in search_results])
            response = self.generate_response(context, query)
            
            print(f"Response: {response}")
            self.text_to_speech(response)
            self.update_history(query, response)

            print("Speak your next query (5 seconds):")
            spoken_query = self.transcribe_microphone(5)
            print(f"Transcribed query: {spoken_query}")

            if spoken_query:
                search_results = self.perform_similarity_search(spoken_query)
                context = " ".join([doc.page_content for doc in search_results])
                response = self.generate_response(context, spoken_query)
                print(f"Response: {response}")
                self.text_to_speech(response)
                self.update_history(spoken_query, response)

    def get_history(self) -> List[Dict[str, List[str]]]:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()

    def save_history(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(self.history, f)

    def load_history(self, filename: str) -> None:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.history = json.load(f)

    def run_interactive(self) -> None:
        self.initialize_components()
        self.load_and_process_documents()
        
        print("Welcome to the Advanced RAG App!")
        print("You can interact using text or voice.")
        print("Type 'voice' to switch to voice input, 'text' to switch back to text input.")
        print("Type 'history' to view interaction history, 'clear' to clear history.")
        print("Type 'quit' to exit the application.")

        input_mode = "text"
        
        while True:
            if input_mode == "text":
                query = input("Enter your query: ")
            else:
                print("Speak your query (5 seconds):")
                query = self.transcribe_microphone(5)
                print(f"Transcribed query: {query}")

            if query.lower() == 'quit':
                break
            elif query.lower() == 'voice':
                input_mode = "voice"
                print("Switched to voice input mode.")
                continue
            elif query.lower() == 'text':
                input_mode = "text"
                print("Switched to text input mode.")
                continue
            elif query.lower() == 'history':
                print("Interaction History:")
                for interaction in self.get_history():
                    print(f"Query: {interaction['query'][0]}")
                    print(f"Response: {interaction['response'][0]}")
                    print("---")
                continue
            elif query.lower() == 'clear':
                self.clear_history()
                print("History cleared.")
                continue

            search_results = self.perform_similarity_search(query)
            context = " ".join([doc.page_content for doc in search_results])
            response = self.generate_response(context, query)
            
            print(f"Response: {response}")
            self.text_to_speech(response)
            self.update_history(query, response)

    def run_batch(self, queries: List[str], output_file: str) -> None:
        self.initialize_components()
        self.load_and_process_documents()

        with open(output_file, 'w') as f:
            for query in queries:
                search_results = self.perform_similarity_search(query)
                context = " ".join([doc.page_content for doc in search_results])
                response = self.generate_response(context, query)
                
                f.write(f"Query: {query}\n")
                f.write(f"Response: {response}\n")
                f.write("---\n")
                
                self.update_history(query, response)

        print(f"Batch processing complete. Results saved to {output_file}")



if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Advanced RAG Application")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the documents")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the language model to use")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive", help="Run mode")
    parser.add_argument("--batch_file", type=str, help="File containing queries for batch mode")
    parser.add_argument("--output_file", type=str, help="Output file for batch mode results")
    args = parser.parse_args()

    app = AdvancedRAGApp(data_dir=args.data_dir, model_name=args.model_name)

    if args.mode == "interactive":
        app.run_interactive()
    elif args.mode == "batch":
        if not args.batch_file or not args.output_file:
            print("Batch mode requires --batch_file and --output_file arguments.")
        else:
            with open(args.batch_file, 'r') as f:
                queries = f.readlines()
            queries = [query.strip() for query in queries if query.strip()]
            app.run_batch(queries, args.output_file)

    # Save history before exiting
    app.save_history("interaction_history.json")