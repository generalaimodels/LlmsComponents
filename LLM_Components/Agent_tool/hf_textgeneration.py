from typing import Dict, List, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.agents import PipelineTool


class TextGenerationTool(PipelineTool):
    default_checkpoint = "distilgpt2"  # Example model, can be changed as per requirement
    description = "A tool for generating text based on the given prompts."
    name = "text_generator"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForCausalLM

    inputs = {
        "prompt": {"type": "text", "description": "The initial text to start generating from."},
        "parameters": {
            "type": "dict",
            "description": "Optional generation parameters like length, temperature, etc."
        }
    }
    output_type = "text"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_model()

    def _initialize_model(self):
        self.tokenizer = self.pre_processor_class.from_pretrained(self.default_checkpoint)
        self.model = self.model_class.from_pretrained(self.default_checkpoint)
        self.model.to(self.device)

    def encode(self, prompt: str, parameters: Dict[str, Union[int, float]] = None) -> Dict[str, torch.Tensor]:
        # Here, encoding is simply tokenizing the input prompt
        if parameters is None:
            parameters = {}
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        return {"input_ids": input_ids, "parameters": parameters}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                max_length=inputs["parameters"].get("max_length", 50),
                temperature=inputs["parameters"].get("temperature", 1.0),
                top_k=inputs["parameters"].get("top_k", 50),
                top_p=inputs["parameters"].get("top_p", 1.0),
                num_return_sequences=inputs["parameters"].get("num_return_sequences", 1),
                do_sample=True,
            )
        return output_sequences

    def decode(self, outputs: torch.Tensor) -> List[str]:
        # Decode the generated tokens into text strings, skipping special tokens
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

    # Optional: You can override this or implement a more sophisticated version in the base class
    def __call__(self, prompt: str, **generate_kwargs):
        encoded = self.encode(prompt, generate_kwargs)
        outputs = self.forward(encoded)
        return self.decode(outputs)
    
