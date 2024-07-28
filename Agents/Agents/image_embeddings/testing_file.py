"""
Similarity Search Module for Text and Images.

This module provides a robust and scalable solution for similarity search
across text and image data using state-of-the-art embedding models.
"""

from typing import List, Dict, Union, Optional, Any, Tuple
from enum import Enum
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    ViTFeatureExtractor,
    ViTModel,
)
from PIL import Image
import requests
from io import BytesIO
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

class DistanceStrategy(Enum):
    """Enum for distance calculation strategies."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"

class ImageTextDataset(Dataset):
    """Custom dataset for handling both image and text data."""
    def __init__(self, data: List[Union[str, Path, Image.Image]], is_image: bool = True):
        self.data = data
        self.is_image = is_image

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[str, Image.Image]:
        item = self.data[idx]
        if self.is_image:
            return item if isinstance(item, Image.Image) else self.load_image(item)
        return item

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Image.Image:
        """Load an image from a file path or URL."""
        if isinstance(image_path, str) and image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content))
        return Image.open(image_path)

def custom_collate(batch: List[Union[str, Image.Image]]) -> List[Union[str, Image.Image]]:
    """Custom collate function to handle both image and text data."""
    return batch

class SimilaritySearch:
    """Class for performing similarity search on text and images."""
    def __init__(
        self,
        image_model: str,
        text_model: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        *model_args: Any,
        **kwargs: Any
    ):
        self.device = device
        self.batch_size = batch_size
        self.image_processor = ViTFeatureExtractor.from_pretrained(image_model, *model_args, **kwargs)
        self.image_model = ViTModel.from_pretrained(image_model, *model_args, **kwargs).to(self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model, *model_args, **kwargs)
        self.text_model = AutoModel.from_pretrained(text_model, *model_args, **kwargs).to(self.device)
        self.distance_strategy = distance_strategy
        self.faiss_index = None

        self.image_transforms = Compose([
            Resize((224, 224)),
            Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Process a single image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.image_transforms(image).unsqueeze(0)  # Add batch dimension

    def process_batch(self, batch: List[Union[str, Image.Image]], is_image: bool) -> torch.Tensor:
        """Process a batch of images or text."""
        if is_image:
            processed_batch = torch.cat([self.process_image(img) for img in batch]).to(self.device)
            with torch.no_grad():
                outputs = self.image_model(processed_batch)
            return outputs.last_hidden_state[:, 0, :]
        else:
            inputs = self.text_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    def build_index(self, data: List[Union[str, Path, Image.Image]], is_image: bool = True):
        """Build the FAISS index from the given data."""
        dataset = ImageTextDataset(data, is_image)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=custom_collate)

        embeddings = []
        documents = []

        for batch in tqdm(dataloader, desc="Processing batches"):
            batch_embeddings = self.process_batch(batch, is_image)
            embeddings.append(batch_embeddings.cpu().numpy())
            documents.extend([Document(page_content=str(item)) for item in batch])

        embeddings = np.vstack(embeddings)

        self.faiss_index = FAISS.from_embeddings(
            embeddings,
            documents,
            embedding_function=lambda x: x,
            distance_strategy="l2" if self.distance_strategy == DistanceStrategy.EUCLIDEAN else "cosine",
        )

    def similarity_search(
        self,
        query: Union[str, Path, Image.Image],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search for the given query."""
        if isinstance(query, (str, Path)) and os.path.isfile(query):
            query = ImageTextDataset.load_image(query)
        
        if isinstance(query, Image.Image):
            query_embedding = self.process_batch([query], is_image=True)
        else:
            query_embedding = self.process_batch([query], is_image=False)

        query_embedding = query_embedding.cpu().numpy()

        return self.faiss_index.similarity_search_with_score_by_vector(
            query_embedding.squeeze(), k=k, filter=filter
        )


def main():
    """Example usage of the SimilaritySearch class."""
    searcher = SimilaritySearch(
        image_model="google/vit-base-patch16-224",
        text_model="bert-base-uncased",
        cache_dir=r"C:\Users\heman\Desktop\components\model"
    )

    # Build index from a folder of images
    image_folder = Path.home() / "Desktop" / "Coding" / "output1" / "image"
    image_paths = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    searcher.build_index(image_paths)

    # Perform similarity search
    query_image = Path.home() / "Desktop" / "Coding" / "output1" / "image" / "amber.png"
    results = searcher.similarity_search(query_image, k=5)

    for doc, score in results:
        print(f"Similar image: {doc.page_content}, Score: {score}")

if __name__ == "__main__":
    main()