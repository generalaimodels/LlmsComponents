import logging
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any, Type
from transformers import (
    PreTrainedModel,
    TFPreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    pipeline,
    Pipeline
)
from transformers.feature_extraction_utils import (
    PreTrainedFeatureExtractor,
  
)
from transformers.image_processing_utils import (
    BaseImageProcessor,
)
import PIL
import numpy
import torch


from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    VideoMAEImageProcessor,
)


from transformers import (
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForTableQuestionAnswering, AutoModelForVisualQuestionAnswering,
    AutoModelForDocumentQuestionAnswering, AutoModelForTokenClassification, AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction, AutoModelForImageClassification, AutoModelForZeroShotImageClassification,
    AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation,
    AutoModelForInstanceSegmentation, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection,
    AutoModelForDepthEstimation, AutoModelForVideoClassification, AutoModelForVision2Seq,
    AutoModelForAudioClassification, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector, AutoModelForTextToSpectrogram, AutoModelForTextToWaveform, AutoBackbone,
    AutoModelForMaskedImageModeling,AutoModel
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


DataType = Union[str, "PIL.Image.Image", "numpy.ndarray", "torch.Tensor"]
class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass

class AgentAIModel:
    model_classes = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "question_answering": AutoModelForQuestionAnswering,
        "table_question_answering": AutoModelForTableQuestionAnswering,
        "visual_question_answering": AutoModelForVisualQuestionAnswering,
        "document_question_answering": AutoModelForDocumentQuestionAnswering,
        "token_classification": AutoModelForTokenClassification,
        "multiple_choice": AutoModelForMultipleChoice,
        "next_sentence_prediction": AutoModelForNextSentencePrediction,
        "image_classification": AutoModelForImageClassification,
        "zero_shot_image_classification": AutoModelForZeroShotImageClassification,
        "image_segmentation": AutoModelForImageSegmentation,
        "semantic_segmentation": AutoModelForSemanticSegmentation,
        "universal_segmentation": AutoModelForUniversalSegmentation,
        "instance_segmentation": AutoModelForInstanceSegmentation,
        "object_detection": AutoModelForObjectDetection,
        "zero_shot_object_detection": AutoModelForZeroShotObjectDetection,
        "depth_estimation": AutoModelForDepthEstimation,
        "video_classification": AutoModelForVideoClassification,
        "vision2seq": AutoModelForVision2Seq,
        "audio_classification": AutoModelForAudioClassification,
        "ctc": AutoModelForCTC,
        "speech_seq2seq": AutoModelForSpeechSeq2Seq,
        "audio_frame_classification": AutoModelForAudioFrameClassification,
        "audio_xvector": AutoModelForAudioXVector,
        "text_to_spectrogram": AutoModelForTextToSpectrogram,
        "text_to_waveform": AutoModelForTextToWaveform,
        "backbone": AutoBackbone,
        "masked_image_modeling": AutoModelForMaskedImageModeling,
        "Automodel": AutoModel,
    }

    @classmethod
    def load_model(cls, model_type: str, model_name_or_path: Union[str, Path], *model_args: Any, **kwargs: Any) -> Type:
        """
        Function to load a transformers model.

        Args:
            model_type (str): The type of the model (e.g., 'causal_lm', 'masked_lm').
            model_name_or_path (Union[str, Path]): The name or path of the model.
            *model_args: Additional positional arguments to pass to the model's from_pretrained method.
            **kwargs: Additional keyword arguments to pass to the model's from_pretrained method.

        Returns:
            model (Type): The loaded model.

        Raises:
            ValueError: If the specified model type is unknown.
            ModelLoadingError: If an error occurs during model loading.

        Examples::
        model_classes = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "question_answering": AutoModelForQuestionAnswering,
        "table_question_answering": AutoModelForTableQuestionAnswering,
        "visual_question_answering": AutoModelForVisualQuestionAnswering,
        "document_question_answering": AutoModelForDocumentQuestionAnswering,
        "token_classification": AutoModelForTokenClassification,
        "multiple_choice": AutoModelForMultipleChoice,
        "next_sentence_prediction": AutoModelForNextSentencePrediction,
        "image_classification": AutoModelForImageClassification,
        "zero_shot_image_classification": AutoModelForZeroShotImageClassification,
        "image_segmentation": AutoModelForImageSegmentation,
        "semantic_segmentation": AutoModelForSemanticSegmentation,
        "universal_segmentation": AutoModelForUniversalSegmentation,
        "instance_segmentation": AutoModelForInstanceSegmentation,
        "object_detection": AutoModelForObjectDetection,
        "zero_shot_object_detection": AutoModelForZeroShotObjectDetection,
        "depth_estimation": AutoModelForDepthEstimation,
        "video_classification": AutoModelForVideoClassification,
        "vision2seq": AutoModelForVision2Seq,
        "audio_classification": AutoModelForAudioClassification,
        "ctc": AutoModelForCTC,
        "speech_seq2seq": AutoModelForSpeechSeq2Seq,
        "audio_frame_classification": AutoModelForAudioFrameClassification,
        "audio_xvector": AutoModelForAudioXVector,
        "text_to_spectrogram": AutoModelForTextToSpectrogram,
        "text_to_waveform": AutoModelForTextToWaveform,
        "backbone": AutoBackbone,
        "masked_image_modeling": AutoModelForMaskedImageModeling,
        "Automodel": AutoModel,
        """
        if model_type not in cls.model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls.model_classes[model_type]

        try:
            model = model_class.from_pretrained(model_name_or_path, *model_args, **kwargs)
        except Exception as e:
            raise ModelLoadingError(f"Error loading model: {e}") from e

        return model
    
  


def AgentPipeline(
    task: str = None,
    model: Optional[Union[str, PreTrainedModel, TFPreTrainedModel]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map: Optional[Union[str, Dict[str, Union[int, str, "torch.device"]]]] = None,
    torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs,
) -> Pipeline:
    """
    Advanced utility factory method to build a robust, optimized, and scalable [`Pipeline`].

    Args:
        task (`str`): The task defining which pipeline will be returned.
        model (`str` or [`PreTrainedModel`] or [`TFPreTrainedModel`], *optional*):
            The model that will be used by the pipeline to make predictions.
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model.
        feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
            The feature extractor that will be used by the pipeline to encode data for the model.
        framework (`str`, *optional*): The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow.
        revision (`str`, *optional*, defaults to `"main"`): The specific model version to use.
        use_fast (`bool`, *optional*, defaults to `True`): Whether or not to use a Fast tokenizer if possible.
        use_auth_token (`str` or *bool*, *optional*): The token to use as HTTP bearer authorization for remote files.
        device (`int` or `str` or `torch.device`): Defines the device on which this pipeline will be allocated.
        device_map (`str` or `Dict[str, Union[int, str, torch.device]`, *optional*):
            Sent directly as `model_kwargs` to specify the device map.
        torch_dtype (`str` or `torch.dtype`, *optional*): Sent directly as `model_kwargs` to use the available precision.
        trust_remote_code (`bool`, *optional*, defaults to `False`): Whether or not to allow for custom code defined on the Hub.
        model_kwargs (`Dict[str, Any]`, *optional*): Additional keyword arguments passed to the model's `from_pretrained` function.
        kwargs (`Dict[str, Any]`, *optional*): Additional keyword arguments passed to the specific pipeline init.

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.
    """
    logger.info(f"Building advanced pipeline for task: {task}")

    # Perform necessary checks and validations
    if device_map and device:
        raise ValueError("Do not use `device_map` and `device` at the same time as they will conflict.")

    # Build the pipeline
    pipeline_test = pipeline(
        task=task,
        model=model,
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

    return pipeline_test



class AgentPreProcessorPipeline:
    """
    An advanced and robust wrapper for various Hugging Face Transformers components.

    This wrapper provides a unified interface for using AutoTokenizer, AutoProcessor,
    AutoImageProcessor, AutoFeatureExtractor, and AutoConfig. It ensures type safety,
    logging, and scalability.

    Attributes:
        model_type (str): The type of model to load (e.g., "bert", "vit").
        pretrained_model_name_or_path (str): The pre-trained model name or path.
        **kwargs: Additional keyword arguments passed to the `from_pretrained` methods.
        
        # if __name__ == "__main__":
               #     text_wrapper = AdvancedDataProcessForHemanth(
               #         "text", "bert-base-uncased", revision="main", 
               #         cache_dir="./model"
               #     ) 
               #     text_data = "This is a test sentence."
               #     processed_text = text_wrapper.process_data(text_data)
               #     print(processed_text)

    """

    def __init__(self, model_type: str, pretrained_model_name_or_path: str, **kwargs):
        self.model_type = model_type
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self._tokenizer = None
        self._processor = None
        self._image_processor = None
        self._feature_extractor = None
        self._config = None
        self.from_pretrained_kwargs = kwargs

        logging.info(
            f"Initializing wrapper for {model_type} with {pretrained_model_name_or_path}"
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        """
        Gets the AutoTokenizer instance.

        Returns:
            AutoTokenizer: The tokenizer object.
        """
        if self._tokenizer is None:
            logging.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, **self.from_pretrained_kwargs
            )
        return self._tokenizer

    @property
    def processor(self) -> AutoProcessor:
        """
        Gets the AutoProcessor instance.

        Returns:
            AutoProcessor: The processor object.
        """
        if self._processor is None:
            logging.info("Loading processor...")
            self._processor = AutoProcessor.from_pretrained(
                self.pretrained_model_name_or_path, **self.from_pretrained_kwargs
            )
        return self._processor

    @property
    def image_processor(self) -> Union[AutoImageProcessor, VideoMAEImageProcessor]:
        """
        Gets the AutoImageProcessor or VideoMAEImageProcessor instance.

        Returns:
            Union[AutoImageProcessor, VideoMAEImageProcessor]: The image processor object.
        """
        if self._image_processor is None:
            logging.info("Loading image processor...")
            if self.model_type == "video":
                self._image_processor = VideoMAEImageProcessor.from_pretrained(
                    self.pretrained_model_name_or_path, **self.from_pretrained_kwargs
                )
            else:
                self._image_processor = AutoImageProcessor.from_pretrained(
                    self.pretrained_model_name_or_path, **self.from_pretrained_kwargs
                )
        return self._image_processor

    @property
    def feature_extractor(self) -> AutoFeatureExtractor:
        """
        Gets the AutoFeatureExtractor instance.

        Returns:
            AutoFeatureExtractor: The feature extractor object.
        """
        if self._feature_extractor is None:
            logging.info("Loading feature extractor...")
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.pretrained_model_name_or_path, **self.from_pretrained_kwargs
            )
        return self._feature_extractor

    @property
    def config(self) -> AutoConfig:
        """
        Gets the AutoConfig instance.

        Returns:
            AutoConfig: The configuration object.
        """
        if self._config is None:
            logging.info("Loading config...")
            self._config = AutoConfig.from_pretrained(
                self.pretrained_model_name_or_path, **self.from_pretrained_kwargs
            )
        return self._config

    def process_data(self):
        """
        Processes input data based on the model type.
        """
        logging.info(f"Processing data for {self.model_type}")

        if self.model_type == "text":
            return self.tokenizer
        elif self.model_type == "image":
            return self.image_processor
        elif self.model_type == "audio":
            return self.feature_extractor
        elif self.model_type == "video":
            return self.image_processor
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
