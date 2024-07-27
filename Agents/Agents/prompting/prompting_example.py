import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto

class PromptElement(Enum):
    ROLE = auto()
    TASK = auto()
    CONTEXT = auto()
    EXAMPLES = auto()
    CONSTRAINTS = auto()
    OUTPUT_FORMAT = auto()
    REASONING_STEPS = auto()

@dataclass
class PromptTemplate:
    role: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None
    examples: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    output_format: Optional[str] = None
    reasoning_steps: Optional[List[str]] = None

    def format(self, **kwargs: Any) -> str:
        """
        Format the prompt template with given parameters.

        Args:
            **kwargs: Dynamic parameters to be inserted into the content.

        Returns:
            str: Formatted prompt string.
        """
        formatted_content = self.content.format(**kwargs, **self.parameters)
        
        prompt_parts: Dict[PromptElement, Union[str, List[str]]] = {
            PromptElement.ROLE: f"Role: {self.role}",
            PromptElement.TASK: f"Task: {formatted_content}",
        }
        
        if self.context:
            prompt_parts[PromptElement.CONTEXT] = f"Context: {self.context}"
        
        if self.examples:
            prompt_parts[PromptElement.EXAMPLES] = ["Examples:"] + [f"- {example}" for example in self.examples]
        
        if self.constraints:
            prompt_parts[PromptElement.CONSTRAINTS] = ["Constraints:"] + [f"- {constraint}" for constraint in self.constraints]
        
        if self.output_format:
            prompt_parts[PromptElement.OUTPUT_FORMAT] = f"Output Format: {self.output_format}"
        
        if self.reasoning_steps:
            prompt_parts[PromptElement.REASONING_STEPS] = ["Reasoning Steps:"] + [f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_steps)]
        
        return self._assemble_prompt(prompt_parts)

    def _assemble_prompt(self, parts: Dict[PromptElement, Union[str, List[str]]]) -> str:
        """
        Assemble the prompt from its constituent parts.

        Args:
            parts: Dictionary of prompt elements and their content.

        Returns:
            str: Assembled prompt string.
        """
        assembled_parts = []
        for element in PromptElement:
            if element in parts:
                content = parts[element]
                if isinstance(content, list):
                    assembled_parts.extend(content)
                else:
                    assembled_parts.append(content)
        
        return "\n\n".join(assembled_parts)

    def to_json(self) -> str:
        """
        Convert the PromptTemplate to a JSON string.

        Returns:
            str: JSON representation of the PromptTemplate.
        """
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'PromptTemplate':
        """
        Create a PromptTemplate instance from a JSON string.

        Args:
            json_str: JSON string representation of a PromptTemplate.

        Returns:
            PromptTemplate: Instance created from the JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)
    
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto


class PromptElement(Enum):
    ROLE = auto()
    TASK = auto()
    CONTENT = auto()
    EXAMPLE = auto()
    CONSTRAINTS = auto()
    OUTPUT_FORMAT = auto()


@dataclass
class PromptTemplate:
    role: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    example: Optional[Union[str, List[str]]] = None
    constraints: Optional[List[str]] = None
    output_format: Optional[str] = None

    def format(self, **kwargs: Any) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Format the prompt template with given parameters.

        Args:
            **kwargs: Dynamic parameters to be inserted into the content.

        Returns:
            List[Dict[str, Union[str, List[str]]]]: Formatted prompt as a list of dictionaries.
        """
        formatted_content = self.content.format(**kwargs, **self.parameters)
        
        prompt_parts: List[Dict[str, Union[str, List[str]]]] = [
            {"role": self.role},
            {"content": formatted_content}
        ]
        
        if self.example:
            prompt_parts.append({"example": self.example if isinstance(self.example, list) else [self.example]})
        
        if self.constraints:
            prompt_parts.append({"constraints": self.constraints})
        
        if self.output_format:
            prompt_parts.append({"output_format": self.output_format})
        
        return prompt_parts

    def to_json(self) -> str:
        """
        Convert the PromptTemplate to a JSON string.

        Returns:
            str: JSON representation of the PromptTemplate.
        """
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'PromptTemplate':
        """
        Create a PromptTemplate instance from a JSON string.

        Args:
            json_str: JSON string representation of a PromptTemplate.

        Returns:
            PromptTemplate: Instance created from the JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)


# Example usage
if __name__ == "__main__":
    template = PromptTemplate(
        role="AI Assistant",
        content="Explain {topic} in simple terms.",
        parameters={"topic": "quantum computing"},
        example="Quantum computing is like a super-fast calculator that can solve complex problems.",
        constraints=["Use simple language", "Avoid technical jargon"],
        output_format="A brief paragraph explanation"
    )

    formatted_prompt = template.format()
    print(json.dumps(formatted_prompt, indent=2))