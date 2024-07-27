   
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
class AgentPromptTemplate:
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
    def from_json(cls, json_str: str) -> 'AgentPromptTemplate':
        """
        Create a PromptTemplate instance from a JSON string.

        Args:
            json_str: JSON string representation of a PromptTemplate.

        Returns:
            PromptTemplate: Instance created from the JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)


# # Example usage
# if __name__ == "__main__":
#     template = AgentPromptTemplate(
#         role="AI Assistant",
#         content="Explain {topic} in simple terms.{depth}",
#         parameters={"topic": "quantum computing", "depth": "details explaination i want"},
#         example="Quantum computing is like a super-fast calculator that can solve complex problems.",
#         constraints=["Use simple language", "Avoid technical jargon"],
#         output_format="A brief paragraph explanation"
#     )

#     formatted_prompt = template.format()
#     print(json.dumps(formatted_prompt, indent=2))