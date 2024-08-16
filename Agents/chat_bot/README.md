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
    git clone lhttp://10.180.30.23:81/root/llms-data.git , than pull Hemanth Branch 
    cd chat_bot
    ```

2. **Install dependencies**:
    ```bash
    conda env create -f environment.yml

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
```bash
python app.py --data_dir "/scratch/hemanth/Hemanth/llms-data/Chat_bot_papers/pdf" --model_name "/scratch/hemanth/LLMs/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693" --mode "interactive"
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


  451  wget https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-1.51.tar.gz
  452  tar -xzf espeak-ng-1.51.tar.gz
  453  cd espeak-ng-1.50
  454  ls -l
  455  cd espeak-ng-1.51
  456  ./configure --prefix=$HOME/local
  457  make
  458  ./configure --prefix=$HOME/local
  459  ./configure --prefix=$HOME/local --enable-compatibility --with-extdict
  460  dpkg -l | grep -E 'espeak|speech'
  461  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  462  ls /usr/lib/x86_64-linux-gnu | grep libespeak-ng
  463  echo $LD_LIBRARY_PATH
  464  pip install --upgrade pyttsx3
  465  cd ..
 
  467  pip uninstall  pyttsx3
  468  pip install -U  pyttsx3

  470  pip install --upgrade pyttsx3
  471  clear
  472  ls -lash
  473  python app.py --data_dir "/scratch/hemanth/Hemanth/llms-data/Chat_bot_papers/pdf" --model_name "/scratch/hemanth/LLMs/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693" --mode "interactive"
  474  ls -lash
  475  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  476  ls /usr/lib/x86_64-linux-gnu
  477  find /usr/lib/x86_64-linux-gnu -name 'libespeak-ng.so.1'
  478  ls -l /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
  479  find /usr/lib/x86_64-linux-gnu -name 'libespeak-ng.so*'
  480  mkdir -p $HOME/lib
  481  cp /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 $HOME/lib/
  482  ln -s $HOME/lib/libespeak-ng.so.1 $HOME/lib/libespeak.so.1
  483  export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH