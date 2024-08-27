import gradio as gr
from typing import List, Union, Optional
import time
import random
from PIL import Image
import numpy as np

def text_generation(input_text: str) -> str:
    """Generate dummy text based on input."""
    try:
        time.sleep(2)  # Simulating processing time
        return f"Generated text based on: {input_text}"
    except Exception as e:
        return f"Error in text generation: {str(e)}"

def translation(audio: Union[str, np.ndarray]) -> str:
    """Translate dummy audio to text."""
    try:
        time.sleep(2)  # Simulating processing time
        return "This is a dummy translation of the audio input."
    except Exception as e:
        return f"Error in translation: {str(e)}"

def text_to_image(text: str) -> Optional[Image.Image]:
    """Generate a dummy image based on input text."""
    try:
        time.sleep(2)  # Simulating processing time
        img = Image.new('RGB', (256, 256), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        return img
    except Exception as e:
        print(f"Error in image generation: {str(e)}")
        return None

def create_interface() -> gr.Blocks:
    """Create and return the Gradio interface."""
    with gr.Blocks(title="Advanced Gradio Interface") as interface:
        gr.Markdown("# Advanced Gradio Interface")

        with gr.Tab("Text Generation"):
            text_input = gr.Textbox(label="Input Text")
            text_output = gr.Textbox(label="Generated Text")
            text_button = gr.Button("Generate Text")

        with gr.Tab("Translation"):
            audio_input = gr.Audio(sources="microphone", type="numpy", label="Audio Input")
            trans_output = gr.Textbox(label="Translated Text")
            trans_button = gr.Button("Translate")

        with gr.Tab("Text to Image"):
            image_text_input = gr.Textbox(label="Input Text for Image")
            image_output = gr.Image(label="Generated Image")
            image_button = gr.Button("Generate Image")

        text_button.click(
            fn=text_generation,
            inputs=text_input,
            outputs=text_output,
            api_name="text_generation"
        )

        trans_button.click(
            fn=translation,
            inputs=audio_input,
            outputs=trans_output,
            api_name="translation"
        )

        image_button.click(
            fn=text_to_image,
            inputs=image_text_input,
            outputs=image_output,
            api_name="text_to_image"
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)