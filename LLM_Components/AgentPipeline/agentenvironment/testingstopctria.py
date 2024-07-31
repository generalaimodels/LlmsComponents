from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch



path="gpt2"
cache_dir="/scratch/hemanth/LLMs/"
token="E:\LLMS\Fine-tuning\data"
model_config = AutoConfig.from_pretrained(path, use_auth_token=token)
bnb_config = BitsAndBytesConfig(
                load_in_8bit=False,
                load_in_4bit=False,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype= torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
model = AutoModelForCausalLM.from_pretrained(
            path,
            from_tf=bool(".ckpt" in path),
            config=model_config,
            use_auth_token=token,
            cache_dir=cache_dir,
            # quantization_config=bnb_config
        )
tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_auth_token=token,
            fast_tokenizer=True,
            cache_dir=cache_dir,
        )
tokenizer.pad_token = tokenizer.eos_token


model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))
max_new_tokens: int = 100,
temperature: float = 0.7,
top_p: float = 0.9,
top_k: int = 50,
num_return_sequences: int = 1,
do_sample: bool = True,

generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,

        }
prompt="what country biggest  in the world"
inputs=tokenizer(prompt, return_tensors="pt")
gen_out=model.generate(**inputs, max_new_tokens= 20)
print("general\n",tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])

gen_out = model.generate(**inputs,max_new_tokens= 150, stop_strings=["Russia"], tokenizer=tokenizer)
print("Russia",tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
gen_out = model.generate(**inputs,max_new_tokens= 150, stop_strings=["india","stop","sex"], tokenizer=tokenizer)
print("complex case",tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])