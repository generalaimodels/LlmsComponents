from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState, prepare_pippy
from accelerate.utils import gather_object

class AdvancedPipelineBatch:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', "meta-llama/Llama-2-7b-chat-hf")
        self.attn_implementation = kwargs.get('attn_implementation', "sdpa")
        self.low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', True)
        self.split_points = kwargs.get('split_points', "auto")
        self.gather_output = kwargs.get('gather_output', False)
        
        self.partial_state = PartialState()
        self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                attn_implementation=self.attn_implementation
            )
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Error initializing model or tokenizer: {str(e)}")

    def prepare_model(self, example_inputs: Dict[str, torch.Tensor]):
        try:
            self.model = prepare_pippy(
                self.model,
                split_points=self.split_points,
                example_kwargs=example_inputs,
                gather_output=self.gather_output
            )
        except Exception as e:
            raise RuntimeError(f"Error preparing model with PiPPy: {str(e)}")

    def tokenize_inputs(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            return inputs.to(0)  # Move to the first GPU
        except Exception as e:
            raise RuntimeError(f"Error tokenizing inputs: {str(e)}")

    def generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        inputs = self.tokenize_inputs(prompts)
        self.prepare_model(inputs)

        try:
            with torch.no_grad():
                output = self.model(**inputs, **generation_kwargs)

            if self.partial_state.is_last_process or self.gather_output:
                next_token_logits = output[0][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                decoded_tokens = self.tokenizer.batch_decode(next_token)
                return gather_object(decoded_tokens) if self.gather_output else decoded_tokens
            return []
        except Exception as e:
            raise RuntimeError(f"Error during generation: {str(e)}")

    def cleanup(self):
        self.partial_state.destroy_process_group()

def main():
    prompts = [
        "I would like to",
        "I really like to",
        "The weather is pretty"
    ]

    try:
        pipeline = AdvancedPipelineBatch(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            split_points="auto",
            gather_output=True
        )

        results = pipeline.generate(prompts)

        if results:
            for prompt, result in zip(prompts, results):
                print(f"Prompt: {prompt}")
                print(f"Next token: {result}\n")
        
        pipeline.cleanup()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()