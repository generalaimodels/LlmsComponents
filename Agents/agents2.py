import os
import json
from datetime import datetime
from typing import List, Dict, Union, Optional

from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Define paths for logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_interaction(agent_name: str, interaction: Dict[str, Union[str, List[str]]]) -> None:
    """Log agent interactions to a file."""
    log_file = os.path.join(LOG_DIR, f"{agent_name}_interactions.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        json.dump(interaction, f)
        f.write("\n")

class Agent:
    """Base class for all agents."""
    def __init__(self, name: str):
        self.name = name

    def handle_prompts(self, prompts: List[str]) -> List[str]:
        """Handle a list of prompts and return responses."""
        raise NotImplementedError("Subclasses must implement this method")

    def log(self, prompt: str, response: str) -> None:
        """Log the interaction."""
        log_interaction(self.name, {'prompt': prompt, 'response': response})

class CodingAgent(Agent):
    """Agent for handling coding-related tasks."""
    def __init__(self):
        super().__init__("coding_agent")
        self.model = pipeline('text-generation', model='gpt-3.5-turbo')

    def handle_prompts(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.model(prompt)[0]['generated_text']
            responses.append(response)
            self.log(prompt, response)
        return responses

class GeneralizedChatbotAgent(Agent):
    """Agent for general conversation tasks."""
    def __init__(self):
        super().__init__("chatbot_agent")
        self.model = pipeline('text-generation', model='gpt-3.5-turbo')

    def handle_prompts(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.model(prompt)[0]['generated_text']
            responses.append(response)
            self.log(prompt, response)
        return responses

class ImageAgent(Agent):
    """Agent for handling image generation tasks."""
    def __init__(self):
        super().__init__("image_agent")
        self.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    def handle_prompts(self, prompts: List[str]) -> List[str]:
        image_paths = []
        for idx, prompt in enumerate(prompts):
            image = self.model(prompt).images[0]
            img_path = os.path.join(LOG_DIR, f"image_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
            image.save(img_path)
            image_paths.append(img_path)
            self.log(prompt, img_path)
        return image_paths

class AgentManager:
    """Manages multiple agents and their interactions."""
    def __init__(self):
        self.agents = {
            'coding': CodingAgent(),
            'chatbot': GeneralizedChatbotAgent(),
            'image': ImageAgent()
        }

    def process_batch(self, batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Process a batch of prompts for different agents."""
        results = {}
        for agent_name, prompts in batch.items():
            if agent_name in self.agents:
                results[agent_name] = self.agents[agent_name].handle_prompts(prompts)
            else:
                results[agent_name] = [f"Error: Unknown agent '{agent_name}'"] * len(prompts)
        return results

def main():
    manager = AgentManager()

    # Example batch of prompts
    batch = {
        'coding': ["Write a Python function to calculate factorial", "Implement a binary search algorithm"],
        'chatbot': ["Tell me about the weather today", "What's the capital of France?"],
        'image': ["A serene landscape with mountains and a lake", "A futuristic cityscape at night"]
    }

    results = manager.process_batch(batch)

    # Print results
    for agent, responses in results.items():
        print(f"{agent.capitalize()} Agent Responses:")
        for response in responses:
            print(f"- {response}")
        print()

if __name__ == "__main__":
    main()