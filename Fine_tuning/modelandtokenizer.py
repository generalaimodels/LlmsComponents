from typing import  Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from finetuningconfig import ModelConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Class for loading a model and its tokenizer."""
    
    def __init__(self, model_config: ModelConfig) -> None:
        """
        Initialize ModelLoader with a ModelConfig instance.

        :param model_config: An instance of ModelConfig with model details.
        """
        self.model_config = model_config

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the pre-trained model and tokenizer.

        :return: A tuple containing the loaded model and tokenizer.
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.pretrained_model_name_or_path,
                *self.model_config.inputs,
                **self.model_config.kwargs
            )
            logger.info("Model loaded successfully.")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.pretrained_model_name_or_path,
                *self.model_config.inputs,
                **self.model_config.kwargs
            )
            logger.info("Tokenizer loaded successfully.")

            return model, tokenizer

        except Exception as e:
            logger.error("Failed to load model and/or tokenizer: %s", e)
            raise ValueError("Could not load model and/or tokenizer.") from e
        
    def get_config(self) -> AutoConfig:
        """
        Retrieve the model configuration.

        :return: An instance of AutoConfig for the model.
        """
        try:
            config = AutoConfig.from_pretrained(
                self.model_config.pretrained_model_name_or_path,
                *self.model_config.inputs,
                **self.model_config.kwargs)
            logger.info("Loaded model configuration successfully.")
            return config
        except Exception as e:
            logger.error("Failed to load model configuration: %s", e)
            raise ValueError("Could not load model configuration.") from e

# # Example usage
# if __name__ == '__main__':
#     try:
#         # Create a model config instance
#         model_config = ModelConfig('gpt2')
#         # Initialize the ModelLoader with the model config
#         model_loader = ModelLoader(model_config)

#         # Load the model and tokenizer
#         model, tokenizer = model_loader.load_model_and_tokenizer()
#         config=model_loader.get_config()

#     except ValueError as e:
#         logger.critical("An error occurred: %s", e)