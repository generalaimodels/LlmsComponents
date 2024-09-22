import math
import os
from typing import List
import logging


def create_video_grid(video_paths: List[str], output_html: str, height: int = 480, width: int = 640) -> None:
    """
    Create an HTML file displaying videos in a grid layout.

    Args:
        video_paths (List[str]): List of video file paths (local or HTTP URLs).
        output_html (str): Output HTML file path.
        height (int, optional): Height of each video. Defaults to 480.
        width (int, optional): Width of each video. Defaults to 640.

    Raises:
        ValueError: If video_paths is empty.
        FileNotFoundError: If a local video file does not exist.
        Exception: If failed to write HTML file.
    """
    import html

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if not video_paths:
        raise ValueError("The list of video paths is empty.")

    # Validate video paths
    logger.debug("Validating video paths.")
    for path in video_paths:
        if not path.startswith(('http://', 'https://')):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Local video file not found: {path}")

    n_videos = len(video_paths)
    logger.debug(f"Number of videos: {n_videos}")

    # Determine grid size (rows and columns)
    cols = math.ceil(n_videos / 2)
    rows = 2 if n_videos > 1 else 1
    logger.debug(f"Grid size: {rows} rows x {cols} columns")

    # Start building HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Grid</title>
        <style>
            .video-grid {
                display: grid;
                grid-template-columns: repeat({cols}, 1fr);
                gap: 10px;
            }
            .video-grid video {{
                width: {width}px;
                height: {height}px;
            }}
        </style>
    </head>
    <body>
        <div class="video-grid">
    """.format(cols=cols, width=width, height=height)

    logger.debug("Building HTML content.")
    # Add video elements to HTML content
    for idx, video_path in enumerate(video_paths):
        safe_video_path = html.escape(video_path)
        video_element = f"""
            <video controls>
                <source src="{safe_video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        """
        html_content += video_element

    # Close HTML tags
    html_content += """
        </div>
    </body>
    </html>
    """

    # Write HTML content to output file
    try:
        logger.debug(f"Writing HTML content to {output_html}")
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        logger.error(f"Failed to write HTML file: {e}")
        raise Exception(f"Failed to write HTML file: {e}")

    logger.info(f"HTML file created successfully: {output_html}")