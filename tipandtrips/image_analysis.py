from typing import List
import os
import math
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def read_image(path: str, h: int = None, w: int = None) -> np.ndarray:
    """
    Read an image from a local path or URL, resize it, and return as a numpy array.
    """
    try:
        if path.startswith('http://') or path.startswith('https://'):
            response = requests.get(path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Local file not found: {path}")
            img = Image.open(path)

        if w and h:
            img = img.resize((w, h), Image.LANCZOS)
        elif w:
            h_size = int((float(img.size[1]) * float(w / float(img.size[0]))))
            img = img.resize((w, h_size), Image.LANCZOS)
        elif h:
            w_size = int((float(img.size[0]) * float(h / float(img.size[1]))))
            img = img.resize((w_size, h), Image.LANCZOS)

        return np.array(img)
    except (requests.HTTPError, FileNotFoundError) as e:
        print(f"Error reading image '{path}': {e}")
        raise
    except Exception as e:
        print(f"Unexpected error reading image '{path}': {e}")
        raise

def create_image_grid(image_paths: List[str], output_html: str, h: int = None, w: int = None) -> None:
    """
    Create an HTML file displaying images in a grid using Plotly.
    """
    n_images = len(image_paths)
    max_rows = 50  # Define a maximum number of rows to keep the grid manageable
    n_cols = max(1, math.ceil(n_images / max_rows))
    n_rows = math.ceil(n_images / n_cols)
    
    # Calculate maximum allowable vertical and horizontal spacing
    epsilon = 1e-6
    max_vertical_spacing = 1 / (n_rows - 1) - epsilon if n_rows > 1 else 1
    vertical_spacing = min(0.02, max_vertical_spacing)

    max_horizontal_spacing = 1 / (n_cols - 1) - epsilon if n_cols > 1 else 1
    horizontal_spacing = min(0.02, max_horizontal_spacing)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )

    valid_images = 0
    for idx, path in enumerate(image_paths):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        try:
            img_array = read_image(path, h=h, w=w)
            fig.add_trace(go.Image(z=img_array), row=row, col=col)
            # Hide axes for the subplot
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)
            valid_images += 1
        except Exception as e:
            print(f"Skipping image '{path}' due to error: {e}")
            continue

    if valid_images == 0:
        raise ValueError("No valid images to display.")

    fig.update_layout(
        height=(h * n_rows if h else 300 * n_rows),
        width=(w * n_cols if w else 300 * n_cols),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.write_html(output_html, auto_open=False)
    fig.write_image("output.png")
    print(f"Image grid saved to '{output_html}'.")


# if __name__ == "__main__":
#     # Example usage
#     image_directory = r"C:\Users\heman\Desktop\Coding\meenakshichaudhary006"
#     output_file = "image_grid.html"
#     height = 20   # Desired image height
#     width = 20    # Desired image width

#     # Collect image paths (local and remote URLs)
#     image_list = [
#         os.path.join(image_directory, img_name)
#         for img_name in os.listdir(image_directory)
#         if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
#     ]

#     # Add any remote image URLs if needed
#     # image_list.extend(['http://example.com/image1.jpg', 'https://example.com/image2.png'])

#     try:
#         create_image_grid(image_list, output_file, h=height, w=width)
#     except Exception as e:
#         print(f"Failed to create image grid: {e}")

from typing import List, Tuple
import os
import sys
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_image_paths(folder_path: str) -> List[str]:
    """
    Retrieve a list of image file paths from the specified folder.

    Args:
        folder_path (str): Path to the image folder.

    Returns:
        List[str]: List of image file paths.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    try:
        files = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(supported_extensions)
        ]
        if not files:
            raise ValueError("No supported image files found in the provided folder.")
        return files
    except FileNotFoundError:
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        raise e


def select_images(image_paths: List[str], grid_size: int) -> List[str]:
    """
    Select a subset of images to fit into the grid.

    Args:
        image_paths (List[str]): List of available image file paths.
        grid_size (int): Number of images per row and column.

    Returns:
        List[str]: Selected image file paths.
    """
    required = grid_size * grid_size
    if len(image_paths) < required:
        raise ValueError(
            f"Not enough images to fill a {grid_size}x{grid_size} grid. "
            f"Required: {required}, Available: {len(image_paths)}"
        )
    return image_paths[:required]


# Determine the correct resampling filter based on Pillow version
if hasattr(Image, 'Resampling'):
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
else:
    RESAMPLE_FILTER = Image.LANCZOS  # For older Pillow versions


def resize_image(image_path: str, size: Tuple[int, int]) -> Image.Image:
    """
    Open and resize an image while maintaining aspect ratio.

    Args:
        image_path (str): Path to the image file.
        size (Tuple[int, int]): Desired size as (width, height).

    Returns:
        Image.Image: Resized PIL Image object.
    """
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size, RESAMPLE_FILTER)
            return img.copy()
    except Exception as e:
        raise IOError(f"Error processing image '{image_path}': {e}")


def process_images(image_paths: List[str], thumb_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Concurrently load and resize images.

    Args:
        image_paths (List[str]): List of image file paths.
        thumb_size (Tuple[int, int]): Desired thumbnail size.

    Returns:
        List[Image.Image]: List of resized PIL Image objects.
    """
    resized_images = []
    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(resize_image, path, thumb_size): path for path in image_paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                img = future.result()
                resized_images.append(img)
            except Exception as exc:
                print(exc, file=sys.stderr)
    if not resized_images:
        raise ValueError("No images were successfully processed.")
    return resized_images


def create_image_grid(images: List[Image.Image], grid_size: int) -> go.Figure:
    """
    Create a Plotly figure with a grid of images.

    Args:
        images (List[Image.Image]): List of PIL Image objects.
        grid_size (int): Number of images per row and column.

    Returns:
        go.Figure: Plotly figure containing the image grid.
    """
    try:
        fig = make_subplots(
            rows=grid_size, cols=grid_size,
            horizontal_spacing=0.01, vertical_spacing=0.01,
            shared_xaxes=True, shared_yaxes=True
        )
        for idx, img in enumerate(images):
            row = (idx // grid_size) + 1
            col = (idx % grid_size) + 1
            # Convert PIL Image to PNG bytes
            img_bytes = img_to_bytes(img)
            fig.add_layout_image(
                dict(
                    source=img_bytes,
                    xref=f"x{col}", yref=f"y{row}",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    sizing="stretch",
                    layer="below"
                )
            )
            # Hide axes
            fig.update_xaxes(showgrid=False, zeroline=False, visible=False, row=row, col=col)
            fig.update_yaxes(showgrid=False, zeroline=False, visible=False, row=row, col=col)
        fig.update_layout(
            width=grid_size * 100, height=grid_size * 100,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create image grid: {e}")


def img_to_bytes(img: Image.Image) -> str:
    """
    Convert a PIL Image to a base64-encoded PNG.

    Args:
        img (Image.Image): PIL Image object.

    Returns:
        str: Base64-encoded image string.
    """
    from io import BytesIO
    import base64

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def save_figure(fig: go.Figure, output_path: str) -> None:
    """
    Save the Plotly figure as a PNG file.

    Args:
        fig (go.Figure): Plotly figure to save.
        output_path (str): Path to save the PNG file.

    """
    try:
        fig.write_image(output_path, format='png')
    except Exception as e:
        raise IOError(f"Failed to save figure to '{output_path}': {e}")


def generate_image_grid(
    folder_path: str, n: int, output_path: str,
    thumb_size: Tuple[int, int] = (50, 50)
) -> None:
    """
    Generate an N x N image grid from images in a folder and save as PNG.

    Args:
        folder_path (str): Path to the folder containing images.
        n (int): Size parameter where grid_size = n // 2.
        output_path (str): Path to save the output grid image.
        thumb_size (Tuple[int, int], optional): Thumbnail size. Defaults to (50, 50).

    """
    start_time = time.time()
    try:
        image_paths = get_image_paths(folder_path)
        grid_size = n // 2
        selected_paths = select_images(image_paths, grid_size)
        images = process_images(selected_paths, thumb_size)
        fig = create_image_grid(images, grid_size)
        save_figure(fig, output_path)
        end_time = time.time()
        elapsed = end_time - start_time
        if elapsed > 2:
            print(f"Warning: Processing took {elapsed:.2f} seconds, which exceeds the 2-second goal.", file=sys.stderr)
        else:
            print(f"Image grid saved to '{output_path}' in {elapsed:.2f} seconds.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


# if __name__ == "__main__":
#     generate_image_grid(
#         folder_path=r"C:\Users\heman\Desktop\Coding\meenakshichaudhary006",
#         output_path="save_image.png",
#         n=10
#     )