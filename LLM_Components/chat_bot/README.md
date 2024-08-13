# Advanced Retrieval-Augmented Generation Application (RAG_APP)

## Overview

This application integrates various advanced components for document loading, splitting, embedding, and searching. It also includes functionality for speech-to-text and text-to-speech, providing a comprehensive solution for handling and querying large document collections. The code follows PEP-8 standards, employs robust and scalable design principles, and avoids hardcoding parameters.

## Features

- **Document Loading**: Load documents from a specified directory using `AdvancedDirectoryLoader`.
- **Document Splitting**: Split large documents into smaller chunks using `AdvancedDocumentSplitter`.
- **Embedding and FAISS Index**: Create embeddings for documents and use FAISS for efficient similarity search.
- **Text Generation**: Generate responses using the `ReadPipeline`.
- **Speech-to-Text**: Transcribe audio files or live microphone input using `SpeechToTextPipeline`.
- **Text-to-Speech**: Convert text to speech using `TextToSpeechAPI`.

## Installation

Ensure you have Python 3.7 or higher installed. Then, install the required Python packages:

```bash
pip install transformers langchain_huggingface faiss-gpu
# Add any other required packages here
```

You may also need to install FFmpeg for audio processing:

```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# For macOS using Homebrew
brew install ffmpeg
```

## Usage

1. **Clone the repository and navigate to the project directory:**

```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Run the main application:**

```bash
python app.py
```

3. **Parameters:**

- `directory`: Path to the directory containing documents.
- `tokenizer_model`: Name of the tokenizer model (e.g., `bert-base-uncased`).
- `chunk_size`: Size of each document chunk.
- `embedding_model`: Name of the embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- `audio_file`: Path to the audio file for transcription testing.
- `duration`: Duration for live microphone transcription in seconds.
- `chunk_length_s`: Chunk length in seconds for streaming transcription.

## Components

### Document Loading

Load documents from a specified directory:

```python
loader = AdvancedDirectoryLoader(directory)
documents = loader.load()
```

### Document Splitting

Split large documents into smaller chunks:

```python
splitter = AdvancedDocumentSplitter(tokenizer=tokenizer, chunk_size=chunk_size)
split_documents = splitter.split_documents(documents)
```

### Embedding and FAISS Index

Create embeddings for documents and use FAISS for efficient similarity search:

```python
embeddings_model = create_embedding_model(model_name)
advanced_faiss = initialize_faiss_index(embeddings_model)
add_documents_to_faiss(advanced_faiss, documents)
results = search_with_faiss(advanced_faiss, query, k)
```

### Speech-to-Text

Transcribe audio files or live microphone input:

```python
pipeline = SpeechToTextPipeline(model_name)
transcription = transcribe_audio(pipeline, audio_file)
mic_transcription = transcribe_microphone(pipeline, duration)
stream_transcriptions = stream_transcription(pipeline, duration, chunk_length_s)
```

### Text-to-Speech

Convert text to speech and save to file:

```python
tts = TextToSpeechAPI()
tts.speak("Hello, this is a test of the text-to-speech API.")
tts.save_to_file("This text will be saved to a file.", "output.mp3", rate=180)
```

## Notes

- Replace `'path/to/audio/file'` with the actual path to your audio file for testing the transcription functionality.
- Customize parameters as needed for your specific use case.

## License

This project is licensed under the MIT License.



