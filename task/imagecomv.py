import gradio as gr
from transformers import pipeline
from typing import Dict, Any, List
import torch
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Initialize pipelines
try:
    depth_estimator = pipeline("depth-estimation", device=DEVICE)
    classifier = pipeline("image-classification", device=DEVICE)
    object_detector = pipeline("object-detection", device=DEVICE)
    segmenter = pipeline("image-segmentation", device=DEVICE)
    text_to_image = pipeline("text-to-image", device=DEVICE)
    image_to_text = pipeline("image-to-text", device=DEVICE)
    image_to_image = pipeline("image-to-image", device=DEVICE)
    # Note: image-to-video and text-to-video are not available in standard transformers pipeline
    unconditional_image_generator = pipeline("unconditional-image-generation", device=DEVICE)
    video_classifier = pipeline("video-classification", device=DEVICE)
    zero_shot_classifier = pipeline("zero-shot-image-classification", device=DEVICE)
    mask_generator = pipeline("mask-generation", device=DEVICE)
    zero_shot_object_detector = pipeline("zero-shot-object-detection", device=DEVICE)
    # Note: text-to-3d and image-to-3d are not available in standard transformers pipeline
    feature_extractor = pipeline("feature-extraction", device=DEVICE)
except Exception as e:
    logger.error(f"Error initializing pipelines: {str(e)}")
    raise

def process_image(image: Image.Image) -> Dict[str, Any]:
    """
    Process the input image through various pipelines.

    Args:
        image (Image.Image): Input image to be processed.

    Returns:
        Dict[str, Any]: Dictionary containing results from various pipelines.
    """
    try:
        results = {
            "Depth Estimation": depth_estimator(image),
            "Image Classification": classifier(image),
            "Object Detection": object_detector(image),
            "Image Segmentation": segmenter(image),
            "Image-to-Text": image_to_text(image),
            "Zero-Shot Image Classification": zero_shot_classifier(image, candidate_labels=["animal", "vehicle", "food"]),
            "Mask Generation": mask_generator(image),
            "Zero-Shot Object Detection": zero_shot_object_detector(image, candidate_labels=["car", "person", "dog"]),
            "Image Feature Extraction": feature_extractor(image)
        }
        return results
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": str(e)}

def text_to_image_generation(prompt: str) -> Image.Image:
    """
    Generate an image from text prompt.

    Args:
        prompt (str): Text prompt for image generation.

    Returns:
        Image.Image: Generated image.
    """
    try:
        return text_to_image(prompt)[0]
    except Exception as e:
        logger.error(f"Error generating image from text: {str(e)}")
        return None

def unconditional_image_generation() -> Image.Image:
    """
    Generate an image unconditionally.

    Returns:
        Image.Image: Generated image.
    """
    try:
        return unconditional_image_generator()[0]
    except Exception as e:
        logger.error(f"Error generating unconditional image: {str(e)}")
        return None

def image_to_image_generation(image: Image.Image) -> Image.Image:
    """
    Generate a new image from an input image.

    Args:
        image (Image.Image): Input image.

    Returns:
        Image.Image: Generated image.
    """
    try:
        return image_to_image(image)[0]
    except Exception as e:
        logger.error(f"Error generating image from image: {str(e)}")
        return None

def video_classification(video_path: str) -> List[Dict[str, float]]:
    """
    Classify a video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        List[Dict[str, float]]: List of classification results.
    """
    try:
        return video_classifier(video_path)
    except Exception as e:
        logger.error(f"Error classifying video: {str(e)}")
        return []

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Task Image Processing")
    
    with gr.Tab("Image Processing"):
        image_input = gr.Image()
        output = gr.JSON()
        image_button = gr.Button("Process Image")
    
    with gr.Tab("Text-to-Image"):
        text_input = gr.Textbox(label="Enter text prompt")
        image_output = gr.Image()
        text_button = gr.Button("Generate Image")
    
    with gr.Tab("Unconditional Image Generation"):
        unconditional_output = gr.Image()
        unconditional_button = gr.Button("Generate Random Image")
    
    with gr.Tab("Image-to-Image"):
        image_to_image_input = gr.Image()
        image_to_image_output = gr.Image()
        image_to_image_button = gr.Button("Generate New Image")
    
    with gr.Tab("Video Classification"):
        video_input = gr.Video()
        video_output = gr.JSON()
        video_button = gr.Button("Classify Video")

    image_button.click(process_image, inputs=image_input, outputs=output)
    text_button.click(text_to_image_generation, inputs=text_input, outputs=image_output)
    unconditional_button.click(unconditional_image_generation, outputs=unconditional_output)
    image_to_image_button.click(image_to_image_generation, inputs=image_to_image_input, outputs=image_to_image_output)
    video_button.click(video_classification, inputs=video_input, outputs=video_output)

if __name__ == "__main__":
    demo.launch()