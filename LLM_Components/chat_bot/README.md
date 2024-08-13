# Advanced RAG Application

The Advanced RAG (Retrieval-Augmented Generation) Application is an interactive tool that leverages advanced document loading, splitting, embedding, and vector search capabilities to enable efficient text retrieval and response generation. It also supports speech-to-text and text-to-speech functionalities for enhanced interactivity.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Interactive Mode](#interactive-mode)
  - [Batch Mode](#batch-mode)
- [Components](#components)
  - [AdvancedDirectoryLoader](#advanceddirectoryloader)
  - [AdvancedDocumentSplitter](#advanceddocumentsplitter)
  - [HuggingFaceEmbeddings](#huggingfaceembeddings)
  - [AdvancedFAISS](#advancedfaiss)
  - [ReadPipeline](#readpipeline)
  - [SpeechToTextPipeline](#speechtotextpipeline)
  - [TextToSpeechAPI](#texttospeechapi)
- [History Management](#history-management)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Document Loading and Splitting**: Efficiently load and split large documents into manageable chunks.
- **Embedding and Vector Search**: Utilize advanced embedding techniques and FAISS for fast and accurate similarity and MMR searches.
- **Text Generation**: Generate context-aware responses using a language model.
- **Speech Recognition and Synthesis**: Convert spoken words to text and synthesize text into speech.
- **History Management**: Save and load interaction histories for review and analysis.

## Installation

1. **Clone the repository**:
    ```bash
    git clone llms-data , than pull Hemanth
    cd chat_bot
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install FAISS**:
    ```bash
    conda install -c pytorch faiss-cpu  # or faiss-gpu for GPU support
    ```

## Usage

The application can be run in two modes: **Interactive** and **Batch**.

### Interactive Mode

In interactive mode, the application allows for a back-and-forth interaction with the user. 

To start the application in interactive mode:

```bash
python app.py --data_dir <path_to_data_dir> --model_name <model_name> --mode interactive
```

### Batch Mode

Batch mode allows the processing of multiple queries from a file and saves the results to an output file.

To run the application in batch mode:

```bash
python app.py --data_dir <path_to_data_dir> --model_name <model_name> --mode batch --batch_file <path_to_batch_file> --output_file <path_to_output_file>
```

## Components

### AdvancedDirectoryLoader

Loads documents from a specified directory.

### AdvancedDocumentSplitter

Splits loaded documents into smaller chunks using a tokenizer.

### HuggingFaceEmbeddings

Generates embeddings using a pre-trained Hugging Face model.

### AdvancedFAISS

Indexes and searches document embeddings using FAISS for similarity and MMR searches.

### ReadPipeline

Generates responses to queries based on the context provided.

### SpeechToTextPipeline

Transcribes speech from audio files or microphone input.

### TextToSpeechAPI

Converts text to speech and plays it back to the user or saves it to a file.

## History Management

The application keeps a history of interactions. You can view, clear, save, or load the interaction history.



