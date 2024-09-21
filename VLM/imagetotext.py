import os
from typing import List, Tuple, Any, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageTextDataset(Dataset):
    """
    A PyTorch Dataset class for loading images and corresponding text descriptions.

    This dataset handles the preprocessing of images and text descriptions,
    including image transformations and text tokenization.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing image paths and text descriptions.
        image_column (str): Name of the column containing image paths.
        text_column (str): Name of the column containing text descriptions.
        transform (Optional[transforms.Compose]): Transformations to be applied to images.
        tokenizer: Tokenizer used to tokenize text descriptions.
        max_length (int): Maximum length for tokenized text.
        padding (str): Padding strategy ('max_length', 'longest', 'do_not_pad').
        truncation (bool): Whether to truncate text to `max_length`.
        special_tokens (bool): Whether to include special tokens in tokenization.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_column: str,
        text_column: str,
        transform: Optional[transforms.Compose] = None,
        tokenizer: Any = None,
        max_length: int = 128,
        padding: str = 'max_length',
        truncation: bool = True,
        special_tokens: bool = True
    ):
        """
        Initializes the ImageTextDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and text descriptions.
            image_column (str): Name of the column containing image paths.
            text_column (str): Name of the column containing text descriptions.
            transform (Optional[transforms.Compose], optional): Transformations to be applied to images.
                Defaults to None.
            tokenizer (Any, optional): Tokenizer used to tokenize text descriptions. Defaults to None.
            max_length (int, optional): Maximum length for tokenized text. Defaults to 128.
            padding (str, optional): Padding strategy. Defaults to 'max_length'.
            truncation (bool, optional): Whether to truncate text to max_length. Defaults to True.
            special_tokens (bool, optional): Whether to include special tokens in tokenization. Defaults to True.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_column = image_column
        self.text_column = text_column
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.special_tokens = special_tokens

        if self.tokenizer is None:
            raise ValueError("A tokenizer must be provided.")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - image_tensor (torch.Tensor): The processed image tensor.
                - input_ids (torch.Tensor): The tokenized text input IDs.
                - attention_mask (torch.Tensor): The attention mask for the input IDs.
        """
        # Get the image path and text description
        image_path = self.dataframe.at[idx, self.image_column]
        text_description = self.dataframe.at[idx, self.text_column]

        # Validate image path
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Default transformation: convert to tensor
            image_tensor = transforms.ToTensor()(image)

        # Tokenize text description
        try:
            encoding = self.tokenizer.encode_plus(
                text_description,
                add_special_tokens=self.special_tokens,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)  # Shape: [sequence_length]
            attention_mask = encoding['attention_mask'].squeeze(0)  # Shape: [sequence_length]
        except Exception as e:
            raise RuntimeError(f"Error tokenizing text: {e}")

        return image_tensor, input_ids, attention_mask


def create_dataloader(
    dataframe: pd.DataFrame,
    image_column: str,
    text_column: str,
    tokenizer: Any,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[transforms.Compose] = None,
    max_length: int = 128,
    padding: str = 'max_length',
    truncation: bool = True,
    special_tokens: bool = True
) -> DataLoader:
    """
    Creates a DataLoader for the ImageTextDataset.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image paths and text descriptions.
        image_column (str): Name of the column containing image paths.
        text_column (str): Name of the column containing text descriptions.
        tokenizer (Any): Tokenizer used to tokenize text descriptions.
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.
        transform (Optional[transforms.Compose], optional): Transformations to be applied to images.
            Defaults to None.
        max_length (int, optional): Maximum length for tokenized text. Defaults to 128.
        padding (str, optional): Padding strategy. Defaults to 'max_length'.
        truncation (bool, optional): Whether to truncate text to max_length. Defaults to True.
        special_tokens (bool, optional): Whether to include special tokens in tokenization. Defaults to True.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = ImageTextDataset(
        dataframe=dataframe,
        image_column=image_column,
        text_column=text_column,
        transform=transform,
        tokenizer=tokenizer,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        special_tokens=special_tokens
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# Example usage:
if __name__ == "__main__":
    import transformers

    # Initialize tokenizer (using HuggingFace tokenizer as an example)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Example DataFrame
    data = {
        'image_paths': [
            '/path/to/image1.jpg',
            '/path/to/image2.jpg',
            '/path/to/image3.jpg'
        ],
        'descriptions': [
            'A description of the first image.',
            'A description of the second image.',
            'A description of the third image.'
        ]
    }
    df = pd.DataFrame(data)

    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add more augmentations if needed
    ])

    # Create DataLoader
    dataloader = create_dataloader(
        dataframe=df,
        image_column='image_paths',
        text_column='descriptions',
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        transform=image_transforms,
        max_length=64,
        padding='max_length',
        truncation=True,
        special_tokens=True
    )

    # Iterate over DataLoader
    for batch in dataloader:
        images, input_ids, attention_masks = batch
        print(f"Images shape: {images.shape}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Masks shape: {attention_masks.shape}")
        # The batch can now be used for model training or inference