import importlib
import inspect
import os
from typing import Dict, Any
import g4f.client as g4f

def generate_markdown_documentation(module_name: str, object_name: str, function_name: str) -> None:
    """
    Generate comprehensive documentation in Markdown format for a given module, object, and function.

    Args:
        module_name (str): The name of the module.
        object_name (str): The name of the object (e.g., class or function) in the module.
        function_name (str): The name of the function within the object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the object or function cannot be found in the module.
    """
    doc_dir = "documentation"
    os.makedirs(doc_dir, exist_ok=True)

    try:
        details = get_module_info(module_name, object_name, function_name)
        
        # Create Markdown documentation
        with open(f"{doc_dir}/{module_name}_{object_name}_{function_name}.md", "w") as f:
            f.write(f"# Documentation for {module_name}.{object_name}.{function_name}\n\n")
            f.write("## Object Documentation\n")
            f.write(f"### Arguments\n```\n{details['object']['arguments']}\n```\n")
            f.write(f"### Docstring\n```\n{details['object']['docstring']}\n```\n")
            f.write(f"### Code\n```python\n{details['object']['code']}\n```\n\n")

            f.write("## Function Documentation\n")
            f.write(f"### Arguments\n```\n{details['function']['arguments']}\n```\n")
            f.write(f"### Docstring\n```\n{details['function']['docstring']}\n```\n")
            f.write(f"### Code\n```python\n{details['function']['code']}\n```\n")

        print(f"Markdown documentation created at {doc_dir}/{module_name}_{object_name}_{function_name}.md")

    except (ImportError, AttributeError) as e:
        print(f"Error: {e}")

def get_module_info(module_name: str, object_name: str, function_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve details about a specific object and function from a given module.

    Args:
        module_name (str): The name of the module.
        object_name (str): The name of the object (e.g., class or function) in the module.
        function_name (str): The name of the function within the object.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the argument details, code, and docstring
                                   of the specified object and function.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the object or function cannot be found in the module.
    """
    try:
        # Import the specified module
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")

    try:
        # Get the specified object from the module
        obj = getattr(module, object_name)
    except AttributeError:
        raise AttributeError(f"Object '{object_name}' not found in module '{module_name}'.")

    try:
        # Get the specified function from the object
        func = getattr(obj, function_name)
    except AttributeError:
        raise AttributeError(f"Function '{function_name}' not found in object '{object_name}'.")

    # Extract function details
    func_signature = inspect.signature(func)
    func_code = inspect.getsource(func)
    func_doc = inspect.getdoc(func)

    # Extract object details
    obj_code = inspect.getsource(obj)
    obj_doc = inspect.getdoc(obj)

    return {
        "object": {
            "arguments": str(func_signature),
            "code": obj_code,
            "docstring": obj_doc or "No docstring available."
        },
        "function": {
            "arguments": str(func_signature),
            "code": func_code,
            "docstring": func_doc or "No docstring available."
        }
    }

def example_usage() -> None:
    """
    Example usage of the module documentation generator.
    """
    module_name = "transformers"
    object_name = "Trainer"
    function_name = "train"

    try:
        generate_markdown_documentation(module_name, object_name, function_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    example_usage()