from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import PartialState
from accelerate.utils import gather_object

class AdvancedBatchPipeline:
    def __init__(self, **kwargs):
        self.distributed_state = PartialState()
        self.model_name = kwargs.get('model_name', "microsoft/phi-2")
        self.batch_size = kwargs.get('batch_size', 2)
        self.pad_to_multiple_of = kwargs.get('pad_to_multiple_of', 8)
        self.max_new_tokens = kwargs.get('max_new_tokens', 20)
        
        self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.distributed_state.device,
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Error initializing model or tokenizer: {str(e)}")

    def _prepare_prompts(self, prompts: List[str]) -> List[List[str]]:
        return [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]

    def _tokenize_prompts(self, formatted_prompts: List[List[str]]) -> List[Dict[str, torch.Tensor]]:
        padding_side_default = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        try:
            tokenized_prompts = [
                self.tokenizer(
                    formatted_prompt,
                    padding=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt"
                ) for formatted_prompt in formatted_prompts
            ]
        finally:
            self.tokenizer.padding_side = padding_side_default
        
        return tokenized_prompts

    def _generate_text(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        batch = batch.to(self.distributed_state.device)
        outputs = self.model.generate(**batch, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def process(self, prompts: List[str], **generation_kwargs) -> List[str]:
        formatted_prompts = self._prepare_prompts(prompts)
        tokenized_prompts = self._tokenize_prompts(formatted_prompts)
        
        completions_per_process = []
        
        try:
            with self.distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
                for batch in batched_prompts:
                    generated_text = self._generate_text(batch)
                    completions_per_process.extend(generated_text)
        except Exception as e:
            raise RuntimeError(f"Error during text generation: {str(e)}")

        completions_gather = gather_object(completions_per_process)
        return completions_gather[:len(prompts)]

def main():
    prompts = [
        "I would like to",
        "hello how are you",
        "what is going on",
        "roses are red and",
        "welcome to the hotel",
    ]

    try:
        pipeline = AdvancedBatchPipeline(
            model_name="microsoft/phi-2",
            batch_size=2,
            pad_to_multiple_of=8,
            max_new_tokens=100
        )
        completions = pipeline.process(prompts)
        
        for prompt, completion in zip(prompts, completions):
            print(f"Prompt: {prompt}")
            print(f"Completion: {completion}\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()