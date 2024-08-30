#!/usr/bin/env python
# coding=utf-8
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from customfinetuning.cconfig import DataConfig, ModelConfig,DatasetConfig
from customfinetuning.cdata import PromptTemplate, DatasetProcessor
from customfinetuning.cdata.data_collector import ConcatDataset_batch,ConcatDataset
from customfinetuning.cmodel.modelandtokenizer import ModelLoader
from training_config import Config,load_config



config = Config.from_yaml(r'E:\LLMS\Fine-tuning\LlmsComponents\Fine_tuning\congfig.yml', 'config1')
print(config.dataconfig,config.modelconfig)
print(config.dataconfig["DataConfig"]) # ModelConfig
print(config.modelconfig["ModelConfig"])
# # Example usage:
user_values = {'do_train': True, 'per_device_train_batch_size': 16}
training_configs = load_config('E:/LLMS/Fine-tuning/LlmsComponents/Fine_tuning/training.yml', user_values)


dataset_config =config.dataconfig["DataConfig"]
data_args=config.dataconfig["DatasetConfig"]
model_config =config.modelconfig["ModelConfig"]
prompt_config =config.promptconfig["PromptTemplate"]


dataset_args=DataConfig(**dataset_config)
data_args=DatasetConfig(**data_args)
model_args=ModelConfig(**model_config)
prompt_template=PromptTemplate(**prompt_config)
print(prompt_template)

model_loader =ModelLoader(model_config=model_args)

try:
    print(model_loader.get_config())
    model, tokenizer = model_loader.load_model_and_tokenizer()
    # tokenizer.bos_token_id = tokenizer.pad_token_id
except Exception as exc:
    raise RuntimeError(f"Failed to load model and tokenizer: {exc}")

dataset_processor = DatasetProcessor(dataloader_config=dataset_args,dataset_config=data_args, tokenizer=tokenizer, prompt_template=prompt_template)
dataset_dict = dataset_processor.process_dataset()



try:
    train_dataset =ConcatDataset_batch(dataset=dataset_dict['train'], chunk_size=512, batch_size=1)
    eval_dataset = ConcatDataset_batch(dataset=dataset_dict['test'], chunk_size=512, batch_size=1)
except KeyError as exc:
    raise RuntimeError(f"Training or testing datasets missing: {exc}")

print(next(iter(train_dataset)))