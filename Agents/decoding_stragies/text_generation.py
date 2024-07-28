from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BitsAndBytesConfigParams = {
    "load_in_8bit": False,
    "load_in_4bit": False,
    "llm_int8_threshold": 6.0,
    "llm_int8_skip_modules": None,
    "llm_int8_enable_fp32_cpu_offload": False,
    "llm_int8_has_fp16_weight": False,
    "bnb_4bit_compute_dtype": None,
    "bnb_4bit_quant_type": "fp4",
    "bnb_4bit_use_double_quant": False,
    "bnb_4bit_quant_storage": None,
}

class DecodingStrategies:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate(self, 
                 input_text: str, 
                 strategies: Optional[Dict[str, Union[bool, float, int, Dict]]] = None,
                 max_length: int = 100, 
                 num_return_sequences: int = 1, 
                 **kwargs) -> List[str]:
        if strategies is None:
            strategies = {}

        inputs = self.tokenizer(input_text, return_tensors="pt")
        generate_kwargs = {
            "max_length": max_length,
            "num_return_sequences": num_return_sequences,
            **kwargs
        }

        # Apply selected strategies
        if "temperature" in strategies:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = strategies["temperature"]
        
        if "top_p" in strategies:
            generate_kwargs["do_sample"] = True
            generate_kwargs["top_p"] = strategies["top_p"]
        
        if "top_k" in strategies:
            generate_kwargs["do_sample"] = True
            generate_kwargs["top_k"] = strategies["top_k"]
        
        if "beam_search" in strategies:
            generate_kwargs["num_beams"] = strategies["beam_search"]
        
        if "diverse_beam_search" in strategies:
            generate_kwargs["num_beams"] = strategies["diverse_beam_search"].get("num_beams", 5)
            generate_kwargs["num_beam_groups"] = strategies["diverse_beam_search"].get("num_beam_groups", 2)
            generate_kwargs["diversity_penalty"] = strategies["diverse_beam_search"].get("diversity_penalty", 1.0)
        
        if "constrained_beam_search" in strategies:
            force_words = strategies["constrained_beam_search"]
            force_words_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in force_words]
            
            def prefix_allowed_tokens_fn(batch_id, input_ids):
                return force_words_ids[batch_id % len(force_words_ids)]
            
            generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        
        if "sequence_to_sequence" in strategies:
            generate_kwargs["encoder_no_repeat_ngram_size"] = 2
            generate_kwargs["no_repeat_ngram_size"] = 2
        
        if "biased_decoding" in strategies:
            bias_words = strategies["biased_decoding"]
            sequence_bias = {
                tuple(self.tokenizer.encode(word, add_special_tokens=False)): score
                for word, score in bias_words.items()
            }
            generate_kwargs["sequence_bias"] = sequence_bias
        
        if "length_penalty" in strategies:
            generate_kwargs["length_penalty"] = strategies["length_penalty"]

        outputs = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Example usage:
decoding_strategies = DecodingStrategies(model_name="gpt2")

prompt_template = [
    {
        "role": "AI Assistant"
    },
    {
        "content": "Explain quantum computing in simple terms.details explanation I want"
    },
    {
        "example": [
            "Quantum computing is like a super-fast calculator that can solve complex problems."
        ]
    },
    {
        "constraints": [
            "Use simple language",
            "Avoid technical jargon"
        ]
    },
    {
        "output_format": "A brief paragraph explanation"
    }
]

# Generate text based on prompt template
strategy = {
    "temperature": 0.7,
    "top_p": 0.9
}

generated_text = decoding_strategies.generate(
    input_text="Explain quantum computing in simple terms. Please provide a detailed explanation, but avoid technical jargon.",
    strategies=strategy,
    max_length=100,
    num_return_sequences=1
)

print(generated_text)



from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class PromptTemplate:
    """A class to handle prompt templates for AI generation."""

    def __init__(self, template: Dict[str, Union[str, List[str]]]):
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


class AdvancedLanguageModel:
    """A class for advanced language model generation with various features."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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


def main():
    # Example usage
    template = {
        "role": "AI Assistant",
        "content": "Explain quantum computing in simple terms.",
        "example": ["Quantum computing is like a super-fast calculator that can solve complex problems."],
        "constraints": ["Use simple language", "Avoid technical jargon"],
        "output_format": "A brief paragraph explanation"
    }
    
    prompt_template = PromptTemplate(template)
    formatted_prompt = prompt_template.format_prompt()
    
    model = AdvancedLanguageModel("gpt2")  # Replace with your preferred model
    generated_texts = model.generate(formatted_prompt, max_length=200, num_return_sequences=3)
    
    for i, text in enumerate(generated_texts, 1):
        print(f"Generated Text {i}:\n{text}\n")


if __name__ == "__main__":
    main()