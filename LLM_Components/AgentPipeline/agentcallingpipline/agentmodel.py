import os
from typing import( 
Union,
List, 
Dict,
Optional,
Callable,
Tuple,
Any
)
import torch
from torch.utils.data import (
Dataset,
IterableDataset, 
DataLoader, 
Dataset
)
from transformers import (
PreTrainedModel,
Trainer,
TrainingArguments,
EvalPrediction,
pipeline,
TrainerCallback,
DataCollator,
BitsAndBytesConfig,
PretrainedConfig,
Pipeline
)
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor
from transformers.models.auto.modeling_auto import AutoModelForDepthEstimation, AutoModelForImageToImage
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from torch import nn
import evaluate
from datasets import load_dataset, DatasetDict
from agentgeneralisedmodel import AgentModel,AgentPreProcessorPipeline




class AgentKnowledgeLearning:
     def __init__(
        self,
        model_type,
        data_type,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs
    ):
       self.model_args=model_args
       self.model_kwargs=kwargs
       self.model=AgentModel.load_model(
                             model_type=model_type,
                             model_name_or_path=pretrained_model_name_or_path,
                             *self.model_args,
                             **self.model_kwargs
        )
       self.tokenizer=AgentPreProcessorPipeline(
                       model_type=data_type,
                       pretrained_model_name_or_path=pretrained_model_name_or_path,
                       *self.model_args,
                       **self.model_kwargs   
        ).process_data()



       
     def generate_text(
        self,
        inputs: Union[List[str], str, List[Dict[str, List[str]]]],
        chat_template: Optional[Union[List[str], str, List[Dict[str, List[str]]]]] = None,
        generation_config: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        if isinstance(inputs, str):
            inputs = [inputs]

        if chat_template:
            if isinstance(chat_template, str):
                chat_template = [chat_template] * len(inputs)
            inputs = [
                self.tokenizer.apply_chat_template(chat, template)
                for chat, template in zip(inputs, chat_template)
            ]

        encoded_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        generation_config = generation_config or {}
        
        with torch.no_grad():
            output_sequences = self.model.generate(**encoded_inputs, **generation_config)

        return self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
     def agent_trainer(
        self,
        args: TrainingArguments = None,
        data_collator: Optional[Union[DataCollator]] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "Dataset"]] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        
    ) -> Trainer:  
        return Trainer(
                       model=self.model,
                       args=args,
                       data_collator=data_collator,
                       train_dataset=train_dataset,
                       eval_dataset=eval_dataset,
                       model_init=model_init,
                       compute_metrics=compute_metrics,
                       callbacks=callbacks,
                       optimizers=optimizers,
                       preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
     
     def AgentTransformerPipeline(
         self,
         task: str = None,
         config: Optional[Union[str, PretrainedConfig]] = None,
         tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
         feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
         image_processor: Optional[Union[str, BaseImageProcessor]] = None,
         framework: Optional[str] = None,
         revision: Optional[str] = None,
         use_fast: bool = True,
         token: Optional[Union[str, bool]] = None,
         device: Optional[Union[int, str, "torch.device"]] = None,
         device_map=None,
         torch_dtype=None,
         trust_remote_code: Optional[bool] = None,
         model_kwargs: Dict[str, Any] = None,
         pipeline_class: Optional[Any] = None,
         **kwargs,
    )->Pipeline:
         
        return pipeline(
                task=task,
                model=self.model,
                config=config,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                image_processor=image_processor,
                framework=framework,
                revision=revision,
                use_fast=use_fast,
                token=token,
                device=device,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                model_kwargs=model_kwargs,
                pipeline_class=pipeline_class,
                **kwargs,
        ) 