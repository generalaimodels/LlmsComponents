from typing import Optional, Union, Dict, Any, List,Iterator
from huggingface_hub import InferenceClient
from PIL import Image
import numpy as np
import json
from pathlib import Path
class AdvancedAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        *,
        token: Union[str, bool, None] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        proxies: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.client = InferenceClient(
            model=model,
            token=token,
            timeout=timeout,
            headers=headers,
            cookies=cookies,
            proxies=proxies,
            base_url=base_url,
            api_key=api_key
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None
    ) -> Union[str, Any]:
        if stream:
            return self._stream_chat_completion(messages, max_tokens)
        return self.client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_format
        )

    def _stream_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int) -> Any:
        return self.client.chat_completion(messages, max_tokens=max_tokens, stream=True)

    def document_question_answering(self, image: Union[str, Image.Image], question: str) -> Any:
        return self.client.document_question_answering(image=image, question=question)

    def feature_extraction(self, text: str) -> np.ndarray:
        return self.client.feature_extraction(text)

    def fill_mask(self, text: str) -> List[Dict[str, Union[float, str]]]:
        return self.client.fill_mask(text)

    def image_classification(self, image: Union[str, Image.Image]) -> List[Dict[str, Union[float, str]]]:
        return self.client.image_classification(image)

    def image_segmentation(self, image: Union[str, Image.Image]) -> List[Dict[str, Any]]:
        return self.client.image_segmentation(image)

    def image_to_image(self, image: Union[str, Image.Image], prompt: str) -> Image.Image:
        return self.client.image_to_image(image, prompt)

    def image_to_text(self, image: Union[str, Image.Image]) -> str:
        return self.client.image_to_text(image)

    def list_deployed_models(self, framework: Optional[str] = None) -> Dict[str, List[str]]:
        return self.client.list_deployed_models(framework)

    def object_detection(self, image: Union[str, Image.Image]) -> List[Dict[str, Any]]:
        return self.client.object_detection(image)

    def question_answering(self, question: str, context: str) -> Dict[str, Union[float, int, str]]:
        return self.client.question_answering(question=question, context=context)

    def sentence_similarity(
        self,
        sentence: str,
        other_sentences: List[str]
    ) -> List[float]:
        return self.client.sentence_similarity(sentence, other_sentences)

    def summarization(self, text: str) -> Dict[str, str]:
        return self.client.summarization(text)

    def table_question_answering(
        self,
        table: Dict[str, List[str]],
        query: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.client.table_question_answering(table, query, model=model)

    def tabular_classification(
        self,
        table: Dict[str, List[str]],
        model: Optional[str] = None
    ) -> List[str]:
        return self.client.tabular_classification(table=table, model=model)
    
    def text_generation(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        stream: bool = False,
        details: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str], Dict[str, Any], Iterator[Dict[str, Any]]]:
        if stream:
            return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, stream=True, details=details, **kwargs)
        return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, details=details, **kwargs)

    def text_generation_with_grammar(
        self,
        prompt: str,
        grammar: Dict[str, Any],
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, grammar=grammar, **kwargs)

    def text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Image.Image:
        return self.client.text_to_image(prompt, negative_prompt=negative_prompt, model=model)

    def text_to_speech(self, text: str) -> bytes:
        return self.client.text_to_speech(text)

    def token_classification(self, text: str) -> List[Dict[str, Any]]:
        return self.client.token_classification(text)

    def translation(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Union[str, Dict[str, str]]:
        return self.client.translation(text, model=model)

    def visual_question_answering(
        self,
        image: Union[str, Image.Image],
        question: str
    ) -> List[Dict[str, Union[float, str]]]:
        return self.client.visual_question_answering(image=image, question=question)

    def zero_shot_classification(
        self,
        text: str,
        labels: List[str],
        multi_label: bool = False,
        hypothesis_template: Optional[str] = None
    ) -> List[Dict[str, Union[str, float]]]:
        return self.client.zero_shot_classification(
            text=text,
            labels=labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template
        )

    def zero_shot_image_classification(
        self,
        image: Union[str, Image.Image],
        labels: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        return self.client.zero_shot_image_classification(image=image, labels=labels)

    def get_endpoint_info(self) -> Dict[str, Any]:
        return self.client.get_endpoint_info()

    def health_check(self) -> bool:
        return self.client.health_check()

    def get_model_status(self, model: str) -> Any:
        return self.client.get_model_status(model)

def main():
    agent = AdvancedAgent(model="meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Example usage
    prompt = "The huggingface_hub library is "
    response = agent.text_generation(prompt, max_new_tokens=12)
    print(f"Text Generation: {response}")

    # Stream example
    print("Streaming Text Generation:")
    for token in agent.text_generation(prompt, max_new_tokens=12, stream=True):
        print(token, end='', flush=True)
    print()

    # Grammar-based generation example
    grammar = {
        "type": "json",
        "value": {
            "properties": {
                "location": {"type": "string"},
                "activity": {"type": "string"},
                "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
                "animals": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["location", "activity", "animals_seen", "animals"],
        },
    }
    grammar_response = agent.text_generation_with_grammar(
        "I saw a puppy a cat and a raccoon during my bike ride in the park",
        grammar=grammar
    )
    print(f"Grammar-based Generation: {json.loads(grammar_response)}")

    # Text to Image example
    image = agent.text_to_image("An astronaut riding a horse on the moon.")
    image.save("astronaut.png")
    print("Image saved as astronaut.png")

    # Text to Speech example
    audio = agent.text_to_speech("Hello world")
    Path("hello_world.flac").write_bytes(audio)
    print("Audio saved as hello_world.flac")

    # Other examples...
    print("Token Classification:", agent.token_classification("My name is Sarah Jessica Parker but you can call me Jessica"))
    print("Translation:", agent.translation("My name is Wolfgang and I live in Berlin"))
    print("Zero-shot Classification:", agent.zero_shot_classification(
        "I really like our dinner and I'm very happy. I don't like the weather though.",
        labels=["positive", "negative", "pessimistic", "optimistic"],
        multi_label=True
    ))
    print("Endpoint Info:", agent.get_endpoint_info())
    print("Health Check:", agent.health_check())
    print("Model Status:", agent.get_model_status("meta-llama/Meta-Llama-3-8B-Instruct"))

if __name__ == "__main__":
    main()    


















from typing import Optional, Union, Dict, Any, List
from huggingface_hub import InferenceClient


class AdvancedAgent:
    def __init__(self, model: Optional[str] = None, *, token: Union[str, bool, None] = None, timeout: Optional[float] = None,
                 headers: Optional[Dict[str, str]] = None, cookies: Optional[Dict[str, str]] = None, proxies: Optional[Any] = None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.client = InferenceClient(model, token=token, timeout=timeout, headers=headers, cookies=cookies,
                                      proxies=proxies, base_url=base_url, api_key=api_key)

    def chat_completion(self, messages: List[Dict[str, Union[str, Any]]], max_tokens: int = 100, response_format: Optional[str] = None, stream: bool = False):
        if stream:
            return self.client.chat_completion(messages, max_tokens=max_tokens, response_format=response_format, stream=stream)
        return self.client.chat_completion(messages, max_tokens=max_tokens, response_format=response_format)

    def document_question_answering(self, image: str, question: str):
        return self.client.document_question_answering(image=image, question=question)

    def feature_extraction(self, text: str):
        return self.client.feature_extraction(text)

    def fill_mask(self, text_with_mask: str):
        return self.client.fill_mask(text_with_mask)

    def image_classification(self, image_url: str):
        return self.client.image_classification(image_url)

    def image_segmentation(self, image_path: str):
        return self.client.image_segmentation(image_path)

    def image_to_image(self, image_path: str, prompt: str):
        return self.client.image_to_image(image_path, prompt)

    def image_to_text(self, image_path_or_url: str):
        return self.client.image_to_text(image_path_or_url)

    def list_deployed_models(self, framework: Optional[str] = None):
        return self.client.list_deployed_models(framework)

    def object_detection(self, image_path: str):
        return self.client.object_detection(image_path)

    def question_answering(self, question: str, context: str):
        return self.client.question_answering(question, context)

    def sentence_similarity(self, sentence: str, other_sentences: List[str]):
        return self.client.sentence_similarity(sentence, other_sentences)

    def summarization(self, text: str):
        return self.client.summarization(text)

    def table_question_answering(self, table: Dict[str, List[str]], query: str, model: Optional[str] = None):
        return self.client.table_question_answering(table, query, model)

    def tabular_classification(self, table: Dict[str, List[str]], model: Optional[str] = None):
        return self.client.tabular_classification(table, model)
    def text_generation(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        stream: bool = False,
        details: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str], Dict[str, Any], Iterator[Dict[str, Any]]]:
        if stream:
            return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, stream=True, details=details, **kwargs)
        return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, details=details, **kwargs)

    def text_generation_with_grammar(
        self,
        prompt: str,
        grammar: Dict[str, Any],
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, grammar=grammar, **kwargs)

    def text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Image.Image:
        return self.client.text_to_image(prompt, negative_prompt=negative_prompt, model=model)

    def text_to_speech(self, text: str) -> bytes:
        return self.client.text_to_speech(text)

    def token_classification(self, text: str) -> List[Dict[str, Any]]:
        return self.client.token_classification(text)

    def translation(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Union[str, Dict[str, str]]:
        return self.client.translation(text, model=model)

    def visual_question_answering(
        self,
        image: Union[str, Image.Image],
        question: str
    ) -> List[Dict[str, Union[float, str]]]:
        return self.client.visual_question_answering(image=image, question=question)

    def zero_shot_classification(
        self,
        text: str,
        labels: List[str],
        multi_label: bool = False,
        hypothesis_template: Optional[str] = None
    ) -> List[Dict[str, Union[str, float]]]:
        return self.client.zero_shot_classification(
            text=text,
            labels=labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template
        )

    def zero_shot_image_classification(
        self,
        image: Union[str, Image.Image],
        labels: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        return self.client.zero_shot_image_classification(image=image, labels=labels)

    def get_endpoint_info(self) -> Dict[str, Any]:
        return self.client.get_endpoint_info()

    def health_check(self) -> bool:
        return self.client.health_check()

    def get_model_status(self, model: str) -> Any:
        return self.client.get_model_status(model)

def main():
    agent = AdvancedAgent(model="meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Example usage
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    response = agent.chat_completion(messages)
    print(response)

if __name__ == "__main__":
    main()
    
