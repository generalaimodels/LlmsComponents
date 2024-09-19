
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import sys


def list_image_paths(directory: str, extensions: List[str] = None) -> List[Path]:
    """
    Recursively list all image files in the given directory with specified extensions.

    Args:
        directory (str): The root directory to search for image files.
        extensions (List[str], optional): List of image file extensions to include. Defaults to common image formats.

    Returns:
        List[Path]: A list of Paths to image files.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    image_paths: List[Path] = []
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise NotADirectoryError(f"The directory '{directory}' does not exist or is not a directory.")

    try:
        for ext in extensions:
            # Use glob to recursively find all files matching the extension
            image_paths.extend(dir_path.rglob(f'*{ext}'))
    except Exception as error:
        print(f"Error while listing image paths: {error}", file=sys.stderr)

    return image_paths


def get_image_size(image_path: Path) -> Tuple[int, int, int]:
    """
    Get the size of an image as (Height, Width, Channels).

    Args:
        image_path (Path): The path to the image file.

    Returns:
        Tuple[int, int, int]: The (Height, Width, Channels) of the image.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode_to_channels = {
                '1': 1,      # (1-bit pixels, black and white, stored with one pixel per byte)
                'L': 1,      # (8-bit pixels, black and white)
                'P': 1,      # (8-bit pixels, mapped to any other mode using a color palette)
                'RGB': 3,    # (3x8-bit pixels, true color)
                'RGBA': 4,   # (4x8-bit pixels, true color with transparency mask)
                'CMYK': 4,   # (4x8-bit pixels, color separation)
                'YCbCr': 3,  # (3x8-bit pixels, color video format)
                'I': 1,      # (32-bit signed integer pixels)
                'F': 1       # (32-bit floating point pixels)
            }
            channels = mode_to_channels.get(img.mode, len(img.getbands()))
            return height, width, channels
    except Exception as error:
        print(f"Error processing image '{image_path}': {error}", file=sys.stderr)
        return 0, 0, 0  # Return zeros to indicate failure in obtaining image size


def main(directory: str):
    """
    Main function to list image paths and print their sizes.

    Args:
        directory (str): The root directory containing images.
    """
    try:
        image_paths = list_image_paths(directory)
        if not image_paths:
            print(f"No images found in directory '{directory}'.")
            return

        for image_path in image_paths:
            height, width, channels = get_image_size(image_path)
            if height > 0 and width > 0 and channels > 0:
                print(f"Image: {image_path} Size: (H: {height}, W: {width}, C: {channels})")
            else:
                print(f"Failed to get size for image: {image_path}", file=sys.stderr)
    except Exception as error:
        print(f"An error occurred: {error}", file=sys.stderr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='List images and their sizes in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing images.')
    args = parser.parse_args()

    main(args.directory)




import os
import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import albumentations as A


def get_image_paths(
    directory: Union[str, Path],
    method: str = 'os',
    file_extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Retrieve a list of image file paths from a specified directory.

    Args:
        directory (Union[str, Path]): Directory to search for images.
        method (str): Method for listing files ('os', 'glob', 'pathlib').
        file_extensions (Optional[List[str]]): List of file extensions to include.

    Returns:
        List[Path]: List of image file paths.

    Raises:
        ValueError: If an invalid method is specified.
        FileNotFoundError: If the directory does not exist.
        RuntimeError: If an error occurs during file listing.
    """
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    image_paths = []

    if method == 'os':
        try:
            for filename in os.listdir(directory):
                file_path = directory / filename
                if file_path.suffix.lower() in file_extensions and file_path.is_file():
                    image_paths.append(file_path)
        except Exception as e:
            raise RuntimeError(f"Error using os.listdir: {e}")
    elif method == 'glob':
        try:
            for ext in file_extensions:
                image_paths.extend(directory.glob(f'*{ext}'))
        except Exception as e:
            raise RuntimeError(f"Error using glob.glob: {e}")
    elif method == 'pathlib':
        try:
            for file_path in directory.iterdir():
                if file_path.suffix.lower() in file_extensions and file_path.is_file():
                    image_paths.append(file_path)
        except Exception as e:
            raise RuntimeError(f"Error using pathlib.Path.iterdir: {e}")
    else:
        raise ValueError(f"Invalid method '{method}'. Valid options are 'os', 'glob', 'pathlib'.")

    return image_paths


def load_image(
    image_path: Union[str, Path],
    loader: str = 'PIL'
) -> Tuple[Union[Image.Image, torch.Tensor], Tuple[int, int, int]]:
    """
    Load an image and obtain its size.

    Args:
        image_path (Union[str, Path]): Path to the image file.
        loader (str): Image loading library to use ('PIL', 'torchvision').

    Returns:
        Tuple[Union[Image.Image, torch.Tensor], Tuple[int, int, int]]: Loaded image and its size (H, W, C).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If an invalid loader is specified.
        RuntimeError: If an error occurs during image loading.
    """
    if not isinstance(image_path, Path):
        image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"The image file '{image_path}' does not exist.")

    if loader == 'PIL':
        try:
            image = Image.open(image_path)
            width, height = image.size
            if image.mode == 'RGB':
                channels = 3
            elif image.mode == 'L':
                channels = 1
            else:
                channels = len(image.getbands())
            size = (height, width, channels)
            return image, size
        except Exception as e:
            raise RuntimeError(f"Error loading image with PIL: {e}")
    elif loader == 'torchvision':
        try:
            image = torchvision.io.read_image(str(image_path))  # (C, H, W)
            channels, height, width = image.shape
            size = (height, width, channels)
            return image, size
        except Exception as e:
            raise RuntimeError(f"Error loading image with torchvision: {e}")
    else:
        raise ValueError(f"Invalid loader '{loader}'. Valid options are 'PIL', 'torchvision'.")


class ImageProcessor:
    """
    A class for processing images with resizing, normalization, and augmentations.
    """

    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = None,
        normalize: Optional[Tuple[List[float], List[float]]] = None,
        augmentations: Optional[List[Callable]] = None,
        library: str = 'torchvision'
    ):
        """
        Initialize the ImageProcessor.

        Args:
            resize (Optional[Tuple[int, int]]): Desired output size (width, height).
            normalize (Optional[Tuple[List[float], List[float]]]): Means and stds for normalization.
            augmentations (Optional[List[Callable]]): List of augmentation functions.
            library (str): Transformation library ('torchvision', 'albumentations').

        Raises:
            ValueError: If an invalid library is specified.
        """
        if library not in ['torchvision', 'albumentations']:
            raise ValueError(f"Invalid library '{library}'. Valid options are 'torchvision', 'albumentations'.")
        self.library = library
        self.resize = resize
        self.normalize = normalize
        self.augmentations = augmentations

        self.transform_pipeline = self._build_transform_pipeline()

    def _build_transform_pipeline(self):
        """
        Build the transformation pipeline based on specified parameters.

        Returns:
            Callable: The composed transformation pipeline.
        """
        if self.library == 'torchvision':
            transforms_list = []
            if self.resize is not None:
                transforms_list.append(T.Resize(self.resize))
            if self.augmentations:
                transforms_list.extend(self.augmentations)
            transforms_list.append(T.ToTensor())
            if self.normalize is not None:
                mean, std = self.normalize
                transforms_list.append(T.Normalize(mean=mean, std=std))
            transform = T.Compose(transforms_list)
        elif self.library == 'albumentations':
            transforms_list = []
            if self.resize is not None:
                transforms_list.append(A.Resize(height=self.resize[1], width=self.resize[0]))
            if self.augmentations:
                transforms_list.extend(self.augmentations)
            if self.normalize is not None:
                mean, std = self.normalize
                transforms_list.append(A.Normalize(mean=mean, std=std))
            transform = A.Compose(transforms_list)
        else:
            raise ValueError(f"Invalid library '{self.library}'. Valid options are 'torchvision', 'albumentations'.")
        return transform

    def process_image(self, image: Union[Image.Image, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply the transformation pipeline to an image.

        Args:
            image (Union[Image.Image, np.ndarray]): Image to process.

        Returns:
            Union[torch.Tensor, np.ndarray]: Processed image.

        Raises:
            RuntimeError: If an error occurs during processing.
        """
        try:
            if self.library == 'torchvision':
                transformed_image = self.transform_pipeline(image)
            elif self.library == 'albumentations':
                if isinstance(image, Image.Image):
                    image = np.array(image)
                transformed = self.transform_pipeline(image=image)
                transformed_image = transformed['image']
            else:
                raise ValueError(f"Invalid library '{self.library}'. Valid options are 'torchvision', 'albumentations'.")
            return transformed_image
        except Exception as e:
            raise RuntimeError(f"Error processing image: {e}")


def process_images(
    image_paths: List[Path],
    image_loader: str = 'PIL',
    processor: Optional[ImageProcessor] = None
) -> List[Union[torch.Tensor, np.ndarray]]:
    """
    Load and process a list of images.

    Args:
        image_paths (List[Path]): List of image file paths.
        image_loader (str): Image loading library to use ('PIL', 'torchvision').
        processor (Optional[ImageProcessor]): ImageProcessor instance to apply transformations.

    Returns:
        List[Union[torch.Tensor, np.ndarray]]: List of processed images.

    Raises:
        RuntimeError: If an error occurs during image processing.
    """
    processed_images = []
    for image_path in image_paths:
        try:
            image, size = load_image(image_path, loader=image_loader)
            if processor is not None:
                processed_image = processor.process_image(image)
            else:
                processed_image = image
            processed_images.append(processed_image)
        except Exception as e:
            print(f"Error processing image '{image_path}': {e}")
            continue
    return processed_images


import os
from pathlib import Path
from typing import List, Callable, Union, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A


def get_image_paths(directory: Union[str, Path], patterns: Optional[List[str]] = None) -> List[Path]:
    """
    Retrieve a list of image file paths from the specified directory matching given patterns.

    Args:
        directory (Union[str, Path]): The directory containing images.
        patterns (Optional[List[str]], optional): List of filename patterns to match.
            Defaults to common image file extensions.

    Returns:
        List[Path]: A list of Paths to image files.

    Raises:
        RuntimeError: If the directory is invalid or an error occurs while accessing files.
    """
    try:
        directory_path = Path(directory)
        if not directory_path.is_dir():
            raise ValueError(f"{directory} is not a valid directory.")

        if patterns is None:
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        image_paths = []
        for pattern in patterns:
            image_paths.extend(directory_path.glob(pattern))
        return image_paths
    except Exception as e:
        raise RuntimeError(f"Error getting image paths from {directory}: {e}") from e


def get_torchvision_transforms(
    resize: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    augmentations: Optional[List[str]] = None
) -> Callable:
    """
    Create a torchvision transformation pipeline.

    Args:
        resize (Optional[Tuple[int, int]], optional): Resize dimensions (Height, Width).
        normalize (bool, optional): Whether to normalize the images.
        augmentations (Optional[List[str]], optional): List of augmentation names to apply.

    Returns:
        Callable: A torchvision transformation pipeline.
    """
    transform_list = []

    if resize is not None:
        transform_list.append(transforms.Resize(resize))

    if augmentations:
        for aug in augmentations:
            if aug == 'RandomHorizontalFlip':
                transform_list.append(transforms.RandomHorizontalFlip())
            elif aug == 'RandomRotation':
                transform_list.append(transforms.RandomRotation(degrees=15))
            elif aug == 'ColorJitter':
                transform_list.append(transforms.ColorJitter())
            # Add more augmentations as needed

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))

    transform = transforms.Compose(transform_list)
    return transform


def get_albumentations_transforms(
    resize: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    augmentations: Optional[List[str]] = None
) -> Callable:
    """
    Create an albumentations transformation pipeline.

    Args:
        resize (Optional[Tuple[int, int]], optional): Resize dimensions (Height, Width).
        normalize (bool, optional): Whether to normalize the images.
        augmentations (Optional[List[str]], optional): List of augmentation names to apply.

    Returns:
        Callable: An albumentations transformation pipeline.
    """
    transform_list = []

    if resize is not None:
        transform_list.append(A.Resize(height=resize[0], width=resize[1]))

    if augmentations:
        for aug in augmentations:
            if aug == 'HorizontalFlip':
                transform_list.append(A.HorizontalFlip(p=0.5))
            elif aug == 'Rotate':
                transform_list.append(A.Rotate(limit=15))
            elif aug == 'ColorJitter':
                transform_list.append(A.ColorJitter())
            # Add more augmentations as needed

    if normalize:
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)))

    transform = A.Compose(transform_list)

    return transform


class CustomImageDataset(Dataset):
    """
    A custom dataset for loading images with optional transformations.

    Attributes:
        image_paths (List[Path]): List of image file paths.
        transform (Optional[Callable]): Transformation function to apply.
        use_albumentations (bool): Flag indicating usage of albumentations transformations.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        transform: Optional[Callable] = None,
        use_albumentations: bool = False
    ):
        """
        Initialize the dataset with image paths and transformations.

        Args:
            image_paths (List[Union[str, Path]]): List of image file paths.
            transform (Optional[Callable], optional): Transformation function to apply.
            use_albumentations (bool, optional): Whether to use albumentations for transformations.
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.use_albumentations = use_albumentations

    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load an image and apply transformations.

        Args:
            idx (int): Index of the image to load.

        Returns:
            torch.Tensor: Transformed image tensor.

        Raises:
            RuntimeError: If an error occurs while processing the image.
        """
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")  # Ensure RGB format

            if self.use_albumentations and self.transform:
                image_np = np.array(image)
                augmented = self.transform(image=image_np)
                image_np = augmented['image']
                image = torch.from_numpy(image_np).permute(2, 0, 1).float()
            else:
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)

            return image
        except Exception as e:
            raise RuntimeError(f"Error processing image {self.image_paths[idx]}: {e}") from e


def main():
    """
    Main function to demonstrate loading images, applying transformations, and creating a DataLoader.
    """
    # Define the image directory (replace with your actual path)
    image_directory = '/path/to/your/images'

    # Get list of image paths
    image_paths = get_image_paths(image_directory)

    # Define transformation parameters
    resize_dims = (224, 224)
    normalize_imgs = True

    augmentations_list = ['RandomHorizontalFlip', 'RandomRotation']  # For torchvision
    # augmentations_list = ['HorizontalFlip', 'Rotate']  # For albumentations

    use_albumentations = False  # Set True to use albumentations

    # Get transformations
    if use_albumentations:
        transform = get_albumentations_transforms(
            resize=resize_dims,
            normalize=normalize_imgs,
            augmentations=augmentations_list
        )
    else:
        transform = get_torchvision_transforms(
            resize=resize_dims,
            normalize=normalize_imgs,
            augmentations=augmentations_list
        )

    # Create dataset
    dataset = CustomImageDataset(
        image_paths=image_paths,
        transform=transform,
        use_albumentations=use_albumentations
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Iterate over DataLoader
    for batch_idx, images in enumerate(dataloader):
        print(f"Batch {batch_idx} - Image batch shape: {images.shape}")
        # Perform further processing
        # For demonstration, we break after the first batch
        break


if __name__ == '__main__':
    main()
