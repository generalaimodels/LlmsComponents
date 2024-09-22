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


if __name__ == "__main__":
    # Example usage
    image_directory = r"C:\Users\heman\Desktop\Coding\meenakshichaudhary006"
    output_file = "image_grid.html"
    height = 20   # Desired image height
    width = 20    # Desired image width

    # Collect image paths (local and remote URLs)
    image_list = [
        os.path.join(image_directory, img_name)
        for img_name in os.listdir(image_directory)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ]

    # Add any remote image URLs if needed
    # image_list.extend(['http://example.com/image1.jpg', 'https://example.com/image2.png'])

    try:
        create_image_grid(image_list, output_file, h=height, w=width)
    except Exception as e:
        print(f"Failed to create image grid: {e}")