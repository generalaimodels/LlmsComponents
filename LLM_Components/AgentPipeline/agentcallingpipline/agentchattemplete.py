from typing import List, Dict, Union, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

class AgentChatTokenizer:
    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        device: Optional[Union[int, str, torch.device]] = None,
        *model_args: Any,
        **kwargs: Any
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = quantization
        self.model_name = model_name
        self.model_args = model_args
        self.kwargs = kwargs
        self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name,*self.model_args,**self.kwargs)
        
        quantization_config = None
        if self.quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=(self.quantization == "8bit"),
                load_in_4bit=(self.quantization == "4bit"),
                llm_int8_enable_fp32_cpu_offload=(self.quantization == "map"),
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype= torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            *self.model_args,
            **self.kwargs
        )

        return tokenizer, model

    def encode(self, chat_data: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, torch.Tensor]:
        prompt = self._format_chat_data(chat_data)
        return self.tokenizer(prompt, return_tensors="pt")

    def decode(self, model_output: torch.Tensor) -> str:
        return self.tokenizer.decode(model_output[0], skip_special_tokens=True)

    def generate_response(
        self,
        chat_data: List[Dict[str, Union[str, List[str]]]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs: Any
    ) -> str:
        input_ids = self.encode(chat_data)
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            **kwargs
        }
        
        output = self.model.generate(**input_ids, **generation_config)
        return self.decode(output)

    def _format_chat_data(self, chat_data: List[Dict[str, Union[str, List[str]]]]) -> str:
        prompt_parts = []
        for item in chat_data:
            if "role" in item:
                prompt_parts.append(f"[{item['role']}]")
            elif "content" in item:
                prompt_parts.append(item['content'])
            elif "example" in item:
                prompt_parts.append("Examples:\n" + "\n".join(f"- {ex}" for ex in item["example"]))
            elif "constraints" in item:
                prompt_parts.append("Constraints:\n" + "\n".join(f"- {con}" for con in item["constraints"]))
            elif "output_format" in item:
                prompt_parts.append(f"Output format: {item['output_format']}")
        return "\n".join(prompt_parts)




def AgentCreatePipeline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: str = "text-generation",
    quantization: Optional[str] = None,
    **kwargs: Any
) -> pipeline:
    quantization_config = None
    if quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=(quantization == "8bit"),
            load_in_4bit=(quantization == "4bit"),
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    return pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_fast=True,
        quantization_config=quantization_config,
        **kwargs
    )



   