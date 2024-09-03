from typing import List
from datacollection import ConvertFolder_to_TxtFolder
from Dataloader import PromptTemplate, DatasetLoader
from rag_config import RagDatasetConfig
from folderloader import DirectoryLoader
import logging
import yaml
import pandas as pd
import logging
# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("RAG pipeline initializing...")

configyml=r"E:\LLMS\Fine-tuning\LlmsComponents\LLM_Components\RAG_pipeline\ragconfig.yml"

with open(configyml, 'r', encoding="utf-8") as file:
    config=yaml.safe_load(file)

pipeline_config=config["config1"]["datasetconfig"]
print(pipeline_config)

if pipeline_config["customdataset"]["customfolder"]:
    ConvertFolder_to_TxtFolder(
       input_folders=[pipeline_config["customdataset"]["input_folder"]],
       output_folder=pipeline_config["customdataset"]["output_folder"]
    )

ragoutputfoldetconfig=RagDatasetConfig(**pipeline_config["RagDatasetConfig"])
dataset=DatasetLoader(ragoutputfoldetconfig).load()

prompt_config=config["config1"]["promptconfig"]["PromptTemplate"]
prompt_template=PromptTemplate(**prompt_config)

data = [
    {
        f"prompt{i}": prompt_template.format(

            text=dataset['text'][i] 
        )
    }
    for i in range(dataset.num_rows)
]

print(data[1]['prompt1'])

