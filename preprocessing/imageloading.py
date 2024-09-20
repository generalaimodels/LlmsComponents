import multiprocessing
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


def _process_image(
    image_path: str,
    channels: int,
    torchvision_transforms: Optional[Callable],
    albumentations_transforms: Optional[Callable],
) -> Optional[Tensor]:
    """
    Load and process a single image.

    Args:
        image_path (str): Path to the image file.
        channels (int): Number of channels to load (1 for grayscale, 3 for RGB).
        torchvision_transforms (Optional[Callable]): Torchvision transforms to apply.
        albumentations_transforms (Optional[Callable]): Albumentations transforms to apply.

    Returns:
        Optional[Tensor]: Processed image tensor, or None if an error occurred.
    """
    try:
        # Load image
        image = Image.open(image_path)
        if channels == 1:
            image = image.convert('L')  # Grayscale
        elif channels == 3:
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")

        # Apply torchvision transforms
        if torchvision_transforms is not None:
            image = torchvision_transforms(image)
        else:
            # Default to conversion to numpy
            image = np.array(image)

        # Convert to numpy array if image is a tensor
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = np.array(image)

        # Apply albumentations transforms
        if albumentations_transforms is not None:
            augmented = albumentations_transforms(image=image_np)
            image_np = augmented['image']

        # Ensure image is tensor
        if isinstance(image_np, np.ndarray):
            image_tensor = torch.from_numpy(image_np)
            if image_tensor.dim() == 3:
                # HWC to CHW
                image_tensor = image_tensor.permute(2, 0, 1)
        else:
            image_tensor = image  # Should already be a tensor

        return image_tensor

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


class ImagePreprocessor:
    """
    Preprocess images with configurable options and multiprocessing.

    Attributes:
        channels (int): Number of channels to load (1 for grayscale, 3 for RGB).
        resize_size (Optional[Tuple[int, int]]): Dimensions to resize images to.
        normalization_mean (Optional[List[float]]): Mean for normalization.
        normalization_std (Optional[List[float]]): Std deviation for normalization.
        torchvision_transforms (Optional[Callable]): Torchvision transforms to apply.
        albumentations_transforms (Optional[Callable]): Albumentations transforms to apply.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of worker processes for multiprocessing.
    """

    def __init__(
        self,
        channels: int = 3,
        resize_size: Optional[Tuple[int, int]] = None,
        normalization_mean: Optional[List[float]] = None,
        normalization_std: Optional[List[float]] = None,
        torchvision_transforms: Optional[Callable] = None,
        albumentations_transforms: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = multiprocessing.cpu_count(),
    ) -> None:
        self.channels = channels
        self.resize_size = resize_size
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Assemble torchvision transforms if not provided
        if torchvision_transforms is None:
            tv_transforms = []
            if self.resize_size is not None:
                tv_transforms.append(transforms.Resize(self.resize_size))
            tv_transforms.append(transforms.ToTensor())
            if (
                self.normalization_mean is not None
                and self.normalization_std is not None
            ):
                tv_transforms.append(
                    transforms.Normalize(
                        mean=self.normalization_mean, std=self.normalization_std
                    )
                )
            self.torchvision_transforms = transforms.Compose(tv_transforms)
        else:
            self.torchvision_transforms = torchvision_transforms

        self.albumentations_transforms = albumentations_transforms

    def process_images(self, image_paths: List[str]) -> List[Tensor]:
        """
        Process a list of images and return batches of tensors.

        Args:
            image_paths (List[str]): List of image file paths.

        Returns:
            List[Tensor]: List of batches of image tensors.
        """
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            func = partial(
                _process_image,
                channels=self.channels,
                torchvision_transforms=self.torchvision_transforms,
                albumentations_transforms=self.albumentations_transforms,
            )
            results = pool.map(func, image_paths)

        # Filter out any None results due to errors
        processed_images = [img for img in results if img is not None]

        if not processed_images:
            print("No images were processed successfully.")
            return []

        # Check all images have same shape
        first_shape = processed_images[0].shape
        for idx, img in enumerate(processed_images):
            if img.shape != first_shape:
                print(
                    f"Image at index {idx} has shape {img.shape}, expected {first_shape}"
                )
                processed_images[idx] = None  # Mark for removal

        # Remove None entries
        processed_images = [img for img in processed_images if img is not None]

        # Pack images into batches
        batches = []
        for i in range(0, len(processed_images), self.batch_size):
            batch_images = processed_images[i : i + self.batch_size]
            try:
                batch = torch.stack(batch_images)
                batches.append(batch)
            except Exception as e:
                print(f"Error stacking batch starting at index {i}: {e}")
        return batches