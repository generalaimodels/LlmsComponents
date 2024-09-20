from typing import Callable, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import PreTrainedTokenizer, AutoTokenizer
import albumentations as A
from torchvision import transforms
import numpy as np


class VisionLanguageDataset(Dataset):
    """
    Dataset class for Vision-Language Models.

    This class handles the loading and processing of images and their corresponding captions.
    Images are processed using torchvision and albumentations transforms.
    Captions are tokenized using a tokenizer from the transformers library.

    Args:
        image_paths (List[str]): List of image file paths.
        captions (List[str]): List of corresponding captions.
        tokenizer (PreTrainedTokenizer): Tokenizer to process captions.
        max_length (int): Maximum length for tokenized captions.
        channels (int): Number of image channels (1 for grayscale, 3 for RGB).
        torchvision_transforms (Optional[Callable]): Torchvision transforms to apply.
        albumentations_transforms (Optional[Callable]): Albumentations transforms to apply.
    """

    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        channels: int = 3,
        torchvision_transforms: Optional[Callable] = None,
        albumentations_transforms: Optional[Callable] = None,
    ) -> None:
        if len(image_paths) != len(captions):
            raise ValueError("Number of image paths and captions must be the same.")

        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.channels = channels
        self.torchvision_transforms = torchvision_transforms
        self.albumentations_transforms = albumentations_transforms

        # Set default transforms if none provided
        if self.torchvision_transforms is None:
            self.torchvision_transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # Process image
        try:
            image = Image.open(image_path)
            if self.channels == 1:
                image = image.convert('L')
            elif self.channels == 3:
                image = image.convert('RGB')
            else:
                raise ValueError(f"Unsupported number of channels: {self.channels}")

            # Apply torchvision transforms
            image = self.torchvision_transforms(image)

            # Convert to numpy for albumentations
            image_np = image.permute(1, 2, 0).numpy()

            # Apply albumentations transforms if any
            if self.albumentations_transforms is not None:
                augmented = self.albumentations_transforms(image=image_np)
                image_np = augmented['image']

                # Convert back to tensor
                image_tensor = torch.from_numpy(image_np)
                if image_tensor.dim() == 3 and image_tensor.shape[2] == self.channels:
                    image_tensor = image_tensor.permute(2, 0, 1)
                else:
                    raise ValueError(
                        f"Unexpected image shape after albumentations: {image_tensor.shape}"
                    )
            else:
                image_tensor = image

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            raise e

        # Tokenize caption
        try:
            encoded_caption = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            input_ids = encoded_caption['input_ids'].squeeze(0)
            attention_mask = encoded_caption['attention_mask'].squeeze(0)

        except Exception as e:
            print(f"Error tokenizing caption '{caption}': {e}")
            raise e

        return image_tensor, input_ids, attention_mask


# Example usage:
if __name__ == "__main__":
    # Sample data
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        # Add more image paths
    ]
    captions = [
        "Caption for image 1.",
        "Caption for image 2.",
        # Add more captions
    ]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Define torchvision transforms
    torchvision_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225]),  # ImageNet stds
    ])

    # Optionally, define albumentations transforms
    albumentations_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        # Add more augmentations as needed
    ])

    # Create dataset
    dataset = VisionLanguageDataset(
        image_paths=image_paths,
        captions=captions,
        tokenizer=tokenizer,
        max_length=128,
        channels=3,
        torchvision_transforms=torchvision_transforms,
        albumentations_transforms=albumentations_transforms,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # Adjust based on your system capabilities
        pin_memory=True,
        drop_last=True,
    )

    # Iterate over DataLoader
    for batch in dataloader:
        images, input_ids, attention_masks = batch

        # Your training code here
        # For example:
        # outputs = model(images, input_ids, attention_masks)
        pass