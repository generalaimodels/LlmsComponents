from datapreprocessing import DatasetProcessor, PromptTemplate
from concatenator import ConcatDatasetBatch
from config import (
    DataproConfig,DatasetConfig,ModelConfig,
    )
from model_tokenizer import (
    ModelLoader
    
    )
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import yaml


path=r"C:\Users\heman\Desktop\Custom_FineTuning\Fine_tuning\config.yml"
with open(path,"r") as file:
    config=yaml.safe_load(file)
  
config=config["config1"]
print(config)   
dataset_args=DatasetConfig(**config['datasetconfig']["DataConfig"])
datasetpro_args=DataproConfig(**config['datasetconfig']["DatasetConfig"])
model_args=ModelConfig(**config['modelconfig']["ModelConfig"])
args_dict=config["promptconfig"]["PromptTemplate"]
print(args_dict)
prompt_template=PromptTemplate(**args_dict)
moder=ModelLoader(model_args)
model, tokenizer=moder.load_model_and_tokenizer()
dataset_dict=DatasetProcessor(
    dataloader_config=dataset_args,
    datasetpro_config=datasetpro_args,
    tokenizer=tokenizer,
    prompt_template=prompt_template).process_dataset()

print(tokenizer.decode(dataset_dict['train'][0]["input_ids"]))
