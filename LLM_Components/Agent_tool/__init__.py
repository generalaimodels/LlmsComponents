"""Agent_tool: A comprehensive toolkit for agent-based operations."""

from typing import List

from .apisearcher import APISearcher
from .command_line_interfacetool import Command_line_interface
from .hf_agenttool import AdvancedPipeline
from .pythoninterceptortool import CodeGenerationSystem
from .searchagent import WebSearchAgent
from .selfprompting import generate_fine_tuned_prompts
from .hf_textgeneration import TextGenerationTool
from .rag import (
    AdvancedDirectoryLoader,
    AdvancedDocumentSplitter,
    AdvancedFAISS,
    RagConfig,
    process_document,
    build_vector_database,
    format_prompt
)

__all__: List[str] = [
    "APISearcher",
    "Command_line_interface",
    "AdvancedPipeline",
    "CodeGenerationSystem",
    "WebSearchAgent",
    "generate_fine_tuned_prompts",
    "TextGenerationTool",
    "AdvancedDirectoryLoader",
    "AdvancedDocumentSplitter",
    "AdvancedFAISS",
    "RagConfig",
    "process_document",
    "build_vector_database",
    "format_prompt"
]

__version__ = "0.1.0"