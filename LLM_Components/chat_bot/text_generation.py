from typing import List, Dict, Union, Optional, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


class ReadPipeline:
    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        *model_args: Any,
        **kwargs: Any
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.model_args = model_args
        self.kwargs = kwargs
        self.tokenizer, self.model = self.load_model_and_tokenizer()
        self.rag_prompt_template = self._create_rag_prompt_template()

    def load_model_and_tokenizer(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, *self.model_args, **self.kwargs)
        
        quantization_config = None
        if self.quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=(self.quantization == "8bit"),
                load_in_4bit=(self.quantization == "4bit"),
                llm_int8_enable_fp32_cpu_offload=(self.quantization == "map"),
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
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

    def _create_rag_prompt_template(self) -> str:
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
            },
        ]
        return self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

    def _format_chat_data(self, context: str, question: str) -> str:
        return self.rag_prompt_template.format(context=context, question=question)

    def encode(self, context: str, question: str) -> Dict[str, torch.Tensor]:
        prompt = self._format_chat_data(context, question)
        return self.tokenizer(prompt, return_tensors="pt")

    def decode(self, model_output: torch.Tensor) -> str:
        return self.tokenizer.decode(model_output[0], skip_special_tokens=True)

    def generate_response(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs: Any
    ) -> str:
        input_ids = self.encode(context, question)
        
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

    def create_pipeline(
        self,
        task: str = "text-generation",
        **kwargs: Any
    ) -> pipeline:
        return pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_fast=True,
            quantization_config=self.quantization,
            **kwargs
        )