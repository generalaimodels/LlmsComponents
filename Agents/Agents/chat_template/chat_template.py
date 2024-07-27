from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


class Chat_tokenizer:
    def __init__(self, tokenizer, model):
        """
        Initialize the Chat_tokenizer with a specified model.

        Args:
            model_name (str): The name of the model to use for tokenization.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def encode(self, chat_data: List[Dict[str, Union[str, List[str]]]]) -> torch.Tensor:
        """
        Encode the chat data into a format suitable for the model.

        Args:
            chat_data (List[Dict]): A list of dictionaries containing chat information.

        Returns:
            torch.Tensor: The encoded input for the model.
        """
        prompt = self._format_chat_data(chat_data)
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return encoded_input

    def decode(self, model_output: torch.Tensor) -> str:
        """
        Decode the model output into human-readable text.

        Args:
            model_output (torch.Tensor): The output tensor from the model.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(model_output[0], skip_special_tokens=True)

    def generate_response(self, chat_data: List[Dict[str, Union[str, List[str]]]]) -> str:
        """
        Generate a response based on the provided chat data.

        Args:
            chat_data (List[Dict]): A list of dictionaries containing chat information.

        Returns:
            str: The generated response.
        """
        input_ids = self.encode(chat_data)
        output = self.model.generate(**input_ids, max_new_tokens=100)
        return self.decode(output)

    def _format_chat_data(self, chat_data: List[Dict[str, Union[str, List[str]]]]) -> str:
        """
        Format the chat data into a prompt string for the model.

        Args:
            chat_data (List[Dict]): A list of dictionaries containing chat information.

        Returns:
            str: A formatted prompt string.
        """
        prompt = ""
        for item in chat_data:
            if "role" in item:
                prompt += f"[{item['role']}]\n"
            elif "content" in item:
                prompt += f"{item['content']}\n"
            elif "example" in item:
                prompt += "Examples:\n" + "\n".join(f"- {ex}" for ex in item["example"]) + "\n"
            elif "constraints" in item:
                prompt += "Constraints:\n" + "\n".join(f"- {con}" for con in item["constraints"]) + "\n"
            elif "output_format" in item:
                prompt += f"Output format: {item['output_format']}\n"
        return prompt.strip()


# Example usage
if __name__ == "__main__":
    chat_tokenizer = Chat_tokenizer()

    chat_data = [
        {"role": "AI Assistant"},
        {"content": "Explain quantum computing in simple terms."},
        {"example": ["Quantum computing is like a super-fast calculator that can solve complex problems."]},
        {"constraints": ["Use simple language", "Avoid technical jargon"]},
        {"output_format": "A brief paragraph explanation"}
    ]

    response = chat_tokenizer.generate_response(chat_data)
    print(response)