from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import numpy as np
import torch
import tensorflow as tf

@dataclass
class PromptTemplate:
    """
    Advanced prompting template with customizable attributes.
    """
    role: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    example: Optional[Union[str, List[str]]] = None
    constraints: Optional[List[str]] = None
    output_format: Optional[str] = None

    def __repr__(self) -> str:
        """
        Custom representation of the PromptTemplate.
        """
        return (f"PromptTemplate(role='{self.role}', content='{self.content}', "
                f"parameters={self.parameters}, example={self.example}, "
                f"constraints={self.constraints}, output_format='{self.output_format}')")

class AdvancedPromptHandler:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_chat_prompt(self, system_prompt: str, context: str, question: str) -> str:
        """
        Create a chat prompt using the given system prompt, context, and question.
        """
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}",
            },
        ]

        return self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Apply chat template to the conversation.
        """
        # Implementation details would go here
        pass

    def encode(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[Union[str, List[str], List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> List[int]:
        """
        Encode the given text(s) using the tokenizer.
        """
        return self.tokenizer.encode(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs
        )

    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Decode the given token IDs back into text.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )



from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from transformers import AutoTokenizer, PaddingStrategy, TruncationStrategy, TensorType


@dataclass
class PromptTemplate:
    role: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    example: Optional[Union[str, List[str]]] = None
    constraints: Optional[List[str]] = None
    output_format: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"PromptTemplate(role={self.role}, content={self.content}, "
            f"parameters={self.parameters}, example={self.example}, "
            f"constraints={self.constraints}, output_format={self.output_format})"
        )


class AdvancedPrompting:
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased"
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], "Conversation"],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Applies the chat template to the given conversation, optionally tokenizing the result.
        """
        formatted_conversation = [
            {
                "role": item.get("role"),
                "content": item.get("content").format(**item.get("parameters", {}))
            }
            for item in (conversation if isinstance(conversation, list) else [conversation])
        ]

        if add_generation_prompt:
            formatted_conversation.append({
                "role": "system",
                "content": "Please generate a response based on the conversation context."
            })

        template_vectorized = self.tokenizer(
            formatted_conversation,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **(tokenizer_kwargs or {}),
            **kwargs
        )

        return template_vectorized if return_dict else template_vectorized.input_ids

    def encode(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs: Any,
    ) -> List[int]:
        """
        Encodes the given text (or text pair) into a list of token IDs.
        """
        return self.tokenizer.encode(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs
        )

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs: Any
    ) -> str:
        """
        Decodes a list of token IDs back into text.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )


# Example of usage:
if __name__ == "__main__":
    system_prompt = "System initialized. How can I help you today?"
    context = "The user is inquiring about the latest advancements in AI."
    question = "What are the newest trends in artificial intelligence?"

    prompt_in_chat_format = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}"}
    ]

    advanced_prompting = AdvancedPrompting()

    result = advanced_prompting.apply_chat_template(
        conversation=prompt_in_chat_format,
        tokenize=False,
        add_generation_prompt=True
    )

    print(result)
    
    
    



from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import numpy as np
import torch
import tensorflow as tf

@dataclass
class PromptTemplate:
    """
    A class representing an advanced prompting template.
    """
    role: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    example: Optional[Union[str, List[str]]] = None
    constraints: Optional[List[str]] = None
    output_format: Optional[str] = None

    def __repr__(self) -> str:
        """
        Return a string representation of the PromptTemplate.
        """
        return (f"PromptTemplate(role='{self.role}', content='{self.content}', "
                f"parameters={self.parameters}, example={self.example}, "
                f"constraints={self.constraints}, output_format={self.output_format})")


class AdvancedPromptManager:
    """
    A class to manage advanced prompting templates and interactions with the tokenizer.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_chat_prompt(self, system_prompt: str, context: str, question: str) -> str:
        """
        Create a chat prompt using the given system prompt, context, and question.
        """
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}",
            },
        ]

        return self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], "Conversation"],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Apply the chat template to the given conversation.
        """
        return self.tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs
        )

    def encode(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[Union[str, List[str], List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> List[int]:
        """
        Encode the given text using the tokenizer.
        """
        return self.tokenizer.encode(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs
        )

    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Decode the given token IDs back into text.
        """
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )