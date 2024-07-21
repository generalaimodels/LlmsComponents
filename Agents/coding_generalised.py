import os
import json
from datetime import datetime
from typing import List, Dict, Union

from transformers import pipeline
from diffusers import StableDiffusionPipeline

# Define paths for logs
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Helper function to log interactions
def log_interaction(agent_name: str, interaction: Dict[str, Union[str, List[str]]]):
    log_file = os.path.join(LOG_DIR, f"{agent_name}_interactions.txt")
    with open(log_file, 'a') as f:
        f.write(json.dumps(interaction) + "\n")

# Define Agents
class CodingAgent:
    def __init__(self):
        self.model = pipeline('text-generation', model='gpt-3.5-turbo')

    def handle_prompts(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.model(prompt)
            responses.append(response[0]['generated_text'])

            # Logging
            log_interaction('coding_agent', {'prompt': prompt, 'response': response[0]['generated_text']})
        return responses

class GeneralizedChatbotAgent:
    def __init__(self):
        self.model = pipeline('text-generation', model='gpt-3.5-turbo')

    def handle_prompts(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.model(prompt)
            responses.append(response[0]['generated_text'])

            # Logging
            log_interaction('chatbot_agent', {'prompt': prompt, 'response': response[0]['generated_text']})
        return responses

# Image Handling with Diffusers
class ImageAgent:
    def __init__(self):
        self.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    def handle_prompts(self, prompts: List[str]) -> List[str]:
        image_paths = []
        for idx, prompt in enumerate(prompts):
            image = self.model(prompt).images[0]
            img_path = os.path.join(LOG_DIR, f"image_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
            image.save(img_path)
            image_paths.append(img_path)

            # Logging
            log_interaction('image_agent', {'prompt': prompt, 'image_path': img_path})
        return image_paths

# Usage Example
if __name__ == "__main__":
    # Instantiate agents
    coding_agent = CodingAgent()
    chatbot_agent = GeneralizedChatbotAgent()
    image_agent = ImageAgent()

    text_prompts = ["Write a Python function to reverse a string.", "Tell me a joke."]
    image_prompts = ["A serene beach sunset.", "A futuristic city skyline."]

    # Get responses
    coding_responses = coding_agent.handle_prompts(text_prompts)
    chatbot_responses = chatbot_agent.handle_prompts(text_prompts)
    image_paths = image_agent.handle_prompts(image_prompts)

    # Print responses
    print("Coding Agent Responses:", coding_responses)
    print("Chatbot Agent Responses:", chatbot_responses)
    print("Generated Image Paths:", image_paths)