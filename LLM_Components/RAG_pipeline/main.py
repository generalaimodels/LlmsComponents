from typing import List
from datacollection import ConvertFolder_to_TxtFolder
from Dataloader import PromptTemplate, DatasetLoader
from rag_config import RagDatasetConfig
import logging
import yaml
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("RAG pipeline initializing...")

# Configuration parameters
CUSTOM_FOLDER: bool = False
USER_FOLDER: List[str] = [""]
OUTPUT_FOLDER: str = ""
MULTIPROCESSING: int = 8
RAG_CONFIG_FILE: str = r"E:\LLMS\Fine-tuning\LlmsComponents\LLM_Components\RAG_pipeline\ragconfig.yml"

# Conditional execution based on CUSTOM_FOLDER flag
if CUSTOM_FOLDER:
    try:
        ConvertFolder_to_TxtFolder(
            input_folders=USER_FOLDER,
            output_folder=OUTPUT_FOLDER,
            max_workers=MULTIPROCESSING
        )
        logger.info("Conversion of folders to text format completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during folder conversion: {str(e)}")

# Load RAG configuration from YAML file
try:
    with open(RAG_CONFIG_FILE, 'r', encoding="utf-8") as file:
        rag_config = yaml.safe_load(file)
        logger.info("RAG configuration loaded successfully.")
except FileNotFoundError:
    logger.error(f"RAG configuration file not found: {RAG_CONFIG_FILE}")
except yaml.YAMLError as e:
    logger.error(f"Error parsing RAG configuration file: {str(e)}")
except Exception as e:
    logger.error(f"An unexpected error occurred while loading the RAG configuration: {str(e)}")

logger.info(f"RAG_Pipeline Argument {rag_config}")


rag_config_main = rag_config["config1"]

# Extract dataset configuration
data_args = rag_config_main.get("datasetconfig", {})
dataset_args = RagDatasetConfig(**data_args.get("RagDatasetConfig", {}))
dataset_c = DatasetLoader(dataset_args)
dataset = dataset_c.load()
logger.info(f"User dataset framework {dataset}")

# Extract prompt configuration
prompt_args = rag_config_main.get("promptconfig", {}).get("PromptTemplate", {})
prompt_template = PromptTemplate(**prompt_args)

# Prepare formatted data with unique prompts
data = [
    {
        f"prompt{i}": prompt_template.format(
            act=dataset['act'][i],
            prompt=dataset['prompt'][i]  # Making each prompt unique
        )
    }
    for i in range(dataset.num_rows)
]



print(data[10]['prompt10'])


