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
import logging


logger = logging.getLogger(__name__)
logger.info(f"Fine Tuning on custom dataset,custom model ")

path=r"E:\LLMS\Fine-tuning\LlmsComponents\LLM_Components\Fine_tuning\config.yml"
with open(path,"r") as file:
    config=yaml.safe_load(file)
  
logger.info(f"This are user arguments { config}")  

config=config["config1"]
dataset_args=DatasetConfig(**config['datasetconfig']["DataConfig"])
datasetpro_args=DataproConfig(**config['datasetconfig']["DatasetConfig"])
model_args=ModelConfig(**config['modelconfig']["ModelConfig"])
args_dict=config["promptconfig"]["PromptTemplate"]
prompt_template=PromptTemplate(**args_dict)
moder=ModelLoader(model_args)
model, tokenizer=moder.load_model_and_tokenizer()
model_config=moder.get_config()
logger.info(f"Model configuration {model_config}")

dataset_dict=DatasetProcessor(
    dataloader_config=dataset_args,
    datasetpro_config=datasetpro_args,
    tokenizer=tokenizer,
    prompt_template=prompt_template).process_dataset()
logger.info(f" We preprocessed stage config we spplit into train and test {dataset_dict}")
print(tokenizer.decode(dataset_dict['train'][0]["input_ids"]))
