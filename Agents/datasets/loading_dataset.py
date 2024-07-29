import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union, Sequence, Mapping

import pandas as pd
from datasets import load_dataset, DatasetDict, DownloadConfig, DownloadMode, Split, VerificationMode
from langchain.docstore.document import Document
from tqdm import tqdm

def load_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Load documents from various file formats in the specified folder.
    """
    documents = []

    for file_path in Path(folder_path).rglob("*"):
        if file_path.is_file():
            if file_path.suffix == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append({"content": f.read(), "source": str(file_path)})
            elif file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    documents.append({"content": " ".join(row.astype(str)), "source": str(file_path)})
            elif file_path.suffix in (".json", ".jsonl",".yml"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        documents.append({"content": json.dumps(data), "source": str(file_path)})
            elif file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
                for _, row in df.iterrows():
                    documents.append({"content": " ".join(row.astype(str)), "source": str(file_path)})

    return documents

def process_dataset(dataset_path: str) -> List[Document]:
    """
    Load and process a dataset, combining all columns into page_content.
    """
    try:
        ds = load_dataset(dataset_path)
        if 'train' not in ds:
            raise KeyError("Dataset does not contain a 'train' split.")

        ds = ds['train']
        columns = ds.column_names

        raw_knowledge_base = []
        for doc in tqdm(ds, desc="Processing documents"):
            page_content = "  ".join(str(doc[col]) for col in columns)
            metadata = {col: str(doc[col]) for col in columns}
            raw_knowledge_base.append(Document(page_content=page_content, metadata=metadata))

        return raw_knowledge_base
    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")
        return []

# Example usage
if __name__ == "__main__":
    folder_path = r"C:\Users\heman\Desktop\components\output"
    dataset_path = r"C:\Users\heman\Desktop\components\output"

    # Load documents from the folder
    documents = load_documents(folder_path)
    print(len(documents))
    # for doc in documents:
    #     print(doc)

    # # Process dataset
    # knowledge_base = process_dataset(dataset_path)
    # # for doc in knowledge_base:
    # #     print(doc)
    # print(len(knowledge_base))