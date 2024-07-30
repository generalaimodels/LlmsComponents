import logging
from typing import Any, Dict, List, Callable, Optional, Tuple,Union
from transformers import AutoModelForCausalLM, AutoTokenizer,set_seed
import torch

class AgentPromptTemplate:
    """A class to handle prompt templates for AI generation."""

    def __init__(self,
                template: Dict[str, Union[str, List[str]]]
                  ):
        self.role: str = template.get("role", "AI Assistant")
        self.content: str = template.get("content", "")
        self.example: List[str] = template.get("example", [])
        self.constraints: List[str] = template.get("constraints", [])
        self.output_format: str = template.get("output_format", "")

    def format_prompt(self) -> str:
        """Format the prompt based on the template."""
        prompt = f"Role: {self.role}\n\n"
        prompt += f"Content: {self.content}\n\n"
        
        if self.example:
            prompt += "Examples:\n"
            prompt += "\n".join(f"- {ex}" for ex in self.example) + "\n\n"
        
        if self.constraints:
            prompt += "Constraints:\n"
            prompt += "\n".join(f"- {con}" for con in self.constraints) + "\n\n"
        
        prompt += f"Output Format: {self.output_format}\n\n"
        prompt += "Response:"
        
        return prompt


class AgentAdvancedLanguageModel:
    """A class for advanced language model generation with various features."""

    def __init__(
            self,
            device,
            model:AutoModelForCausalLM,
            tokenizer:AutoTokenizer,
            Seed:int=42,
            ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        set_seed(seed=Seed)
        
    def generate(self, 
                 prompt: str, 
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 top_k: int = 50,
                 num_return_sequences: int = 1,
                 **kwargs) -> List[str]:
        """Generate text based on the given prompt and parameters."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                **kwargs
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
