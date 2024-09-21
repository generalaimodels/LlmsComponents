
import os
import csv
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class VisionLanguageDataset(Dataset):
    """Custom Dataset for loading and processing images and corresponding text."""

    def __init__(
        self,
        csv_file: str,
        image_transform: transforms.Compose = None,
        tokenizer_name: str = "bert-base-uncased",
        max_text_length: int = 128,
    ):
        """
        Initializes the VisionLanguageDataset.

        Args:
            csv_file (str): Path to the CSV file with image paths and text descriptions.
            image_transform (transforms.Compose, optional): Transformations to apply to images.
            tokenizer_name (str): Name of the tokenizer to use from Hugging Face Transformers.
            max_text_length (int): Maximum length of tokenized text sequences.
        """
        self.image_paths, self.texts = self._load_csv(csv_file)
        self.image_transform = image_transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_length = max_text_length

    @staticmethod
    def _load_csv(csv_file: str) -> Tuple[List[str], List[str]]:
        """
        Loads image paths and texts from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            Tuple[List[str], List[str]]: Lists of image paths and corresponding texts.
        """
        image_paths = []
        texts = []
        try:
            with open(csv_file, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    image_path = row.get("Image")  # Adjust column name as necessary
                    text = row.get("Description")  # Adjust column name as necessary
                    if image_path and text:
                        image_paths.append(image_path)
                        texts.append(text)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
        except Exception as exc:
            raise Exception(f"Error reading CSV file '{csv_file}': {exc}") from exc
        return image_paths, texts

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the image and text at the specified index after applying transformations.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            dict: Dictionary containing the processed image and tokenized text.
        """
        image_path = self.image_paths[idx]
        text = self.texts[idx]

        # Load and process the image
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if self.image_transform:
                    img = self.image_transform(img)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        except Exception as exc:
            raise Exception(f"Error loading image '{image_path}': {exc}") from exc

        # Tokenize the text
        try:
            encoded_text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
        except Exception as exc:
            raise Exception(f"Error tokenizing text: {exc}") from exc

        sample = {
            "image": img,
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
        }
        return sample


# if __name__ == "__main__":
#     # Define image transformations
#     image_transforms = transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#             ),
#         ]
#     )

#     # Initialize the dataset
#     dataset = VisionLanguageDataset(
#         csv_file="data.csv",
#         image_transform=image_transforms,
#         tokenizer_name="bert-base-uncased",
#         max_text_length=128,
#     )

#     # Prepare the DataLoader
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=32, shuffle=True, num_workers=4
#     )

#     # Example iteration over the DataLoader
#     for batch in dataloader:
#         images = batch["image"]
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         # Model processing would go here
#         pass






import os
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_augment_audio(audio_path: str, augment: bool = True) -> Optional[np.ndarray]:
    """
    Load an audio file and perform augmentation.

    Args:
        audio_path (str): Path to the audio file.
        augment (bool): Whether to perform augmentation.

    Returns:
        Optional[np.ndarray]: Audio time series data, or None if loading fails.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        if augment:
            audio = augment_audio(audio, sr)
        return audio
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        return None


def augment_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Perform audio augmentation.

    Args:
        audio (np.ndarray): Audio time series data.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Augmented audio data.
    """
    audio_aug = audio.copy()

    # Add random noise
    noise = np.random.randn(len(audio_aug))
    audio_aug = audio_aug + 0.005 * noise

    # Pitch shift
    n_steps = np.random.uniform(-2, 2)
    audio_aug = librosa.effects.pitch_shift(audio_aug, sr, n_steps)

    # Time stretch
    rate = np.random.uniform(0.9, 1.1)
    audio_aug = librosa.effects.time_stretch(audio_aug, rate)

    return audio_aug


def tokenize_text(text: str, tokenizer) -> List[int]:
    """
    Tokenize text into tokens suitable for the language model.

    Args:
        text (str): Input text.
        tokenizer: Tokenizer instance from transformers library.

    Returns:
        List[int]: Token IDs.
    """
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return []


def process_data(csv_file: str, tokenizer) -> List[Tuple[np.ndarray, List[int]]]:
    """
    Process the data from the CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        tokenizer: Tokenizer instance from transformers library.

    Returns:
        List[Tuple[np.ndarray, List[int]]]: List of tuples containing audio data and tokenized text.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file}: {e}")
        return []

    data = []

    for index, row in df.iterrows():
        audio_path = row['Path of Audio(Audio)']
        text = row['details description of text in great detail']

        if not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            continue

        audio_data = load_and_augment_audio(audio_path)
        if audio_data is None:
            continue

        tokenized_text = tokenize_text(text, tokenizer)
        if not tokenized_text:
            continue

        data.append((audio_data, tokenized_text))

    return data


class AudioTextDataset(Dataset):
    """
    Dataset for audio and text data.
    """

    def __init__(self, data: List[Tuple[np.ndarray, List[int]]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_data, tokenized_text = self.data[idx]

        # Convert audio data to tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        # Convert tokenized text to tensor
        text_tensor = torch.tensor(tokenized_text, dtype=torch.long)

        return audio_tensor, text_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to batch variable-length sequences.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): List of (audio_tensor, text_tensor).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Padded audio tensors
            - Audio lengths
            - Padded text tensors
            - Text lengths
    """
    audio_tensors = [item[0] for item in batch]
    text_tensors = [item[1] for item in batch]

    # Pad audio sequences
    audio_lengths = torch.tensor([len(a) for a in audio_tensors], dtype=torch.long)
    padded_audio = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True)

    # Pad text sequences
    text_lengths = torch.tensor([len(t) for t in text_tensors], dtype=torch.long)
    padded_text = torch.nn.utils.rnn.pad_sequence(text_tensors, batch_first=True)

    return padded_audio, audio_lengths, padded_text, text_lengths


if __name__ == '__main__':
    # Replace 'bert-base-uncased' with the appropriate tokenizer for your language model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    csv_file = 'path_to_csv_file.csv'  # Replace with your CSV file path
    data = process_data(csv_file, tokenizer)
    dataset = AudioTextDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        audio_batch, audio_lengths, text_batch, text_lengths = batch
        # Now feed this batch to your model