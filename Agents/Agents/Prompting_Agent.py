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
            prompt_parts[PromptElement.REASONING_STEPS] = ["Reasoning Steps:"] + [f"{i+1}. {step}" for i, step in enumerate(self.reasoning_steps)]
        
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


def create_advanced_python_template() -> PromptTemplate:
    """
    Create an advanced Python coding prompt template.

    Returns:
        PromptTemplate: An instance of PromptTemplate for advanced Python coding tasks.
    """
    return PromptTemplate(
        role="Advanced Python Developer",
        content="Implement a {task} using advanced Python techniques. The solution should be efficient, readable, and follow best practices.",
        parameters={"complexity": "high"},
        context="You are working on a large-scale Python project that requires optimal performance and maintainability.",
        examples=[
            "Implement a concurrent web scraper using asyncio and aiohttp",
            "Create a memory-efficient data processing pipeline using generators and itertools"
        ],
        constraints=[
            "Use type hints for all function signatures",
            "Implement proper error handling and logging",
            "Optimize for both time and space complexity",
            "Follow PEP 8 style guidelines"
        ],
        output_format="Provide the Python code along with brief comments explaining key design decisions and any advanced techniques used.",
        reasoning_steps=[
            "Analyze the problem and break it down into smaller components",
            "Consider potential performance bottlenecks and optimization strategies",
            "Choose appropriate data structures and algorithms for each component",
            "Implement the solution with a focus on readability and maintainability",
            "Refactor and optimize the code, ensuring it adheres to all constraints",
            "Add comprehensive error handling and logging  ",
            # "Document the code and explain key design decisions"
        ]
    )


