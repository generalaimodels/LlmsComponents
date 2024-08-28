import importlib
import inspect
from typing import Dict, Any, Optional
from pathlib import Path
import os
import markdown
from g4f.client import Client

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
        ImportError: If the specified module is not found.
        AttributeError: If the specified object or function is not found.
    """
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, object_name)
        func = getattr(obj, function_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")
    except AttributeError as e:
        raise AttributeError(f"{e}. Check if '{object_name}' and '{function_name}' exist in '{module_name}'.")

    func_signature = inspect.signature(func)
    func_code = inspect.getsource(func)
    func_doc = inspect.getdoc(func) or "No docstring available."

    obj_code = inspect.getsource(obj)
    obj_doc = inspect.getdoc(obj) or "No docstring available."

    return {
        "object": {
            "arguments": str(func_signature),
            "code": obj_code,
            "docstring": obj_doc
        },
        "function": {
            "arguments": str(func_signature),
            "code": func_code,
            "docstring": func_doc
        }
    }

def generate_documentation(module_info: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Generate comprehensive documentation for the given module information.

    Args:
        module_info (Dict[str, Dict[str, Any]]): The module information dictionary.
        output_dir (Path): The directory to save the generated documentation.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Markdown files
    for item_type in ["object", "function"]:
        md_content = f"# {item_type.capitalize()} Documentation\n\n"
        md_content += f"## Arguments\n```\n{module_info[item_type]['arguments']}\n```\n\n"
        md_content += f"## Code\n```python\n{module_info[item_type]['code']}\n```\n\n"
        md_content += f"## Docstring\n{module_info[item_type]['docstring']}\n\n"

        with open(output_dir / f"{item_type}_documentation.md", "w") as f:
            f.write(md_content)

    # Generate enhanced docstrings
    client = Client()
    for item_type in ["object", "function"]:
        prompt = f"Enhance the following docstring for a Python {item_type}:\n\n{module_info[item_type]['docstring']}\n\nProvide a more detailed explanation, include argument descriptions, and add usage examples."
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        enhanced_docstring = response.choices[0].message.content

        # Update the code with enhanced docstring
        updated_code = module_info[item_type]['code'].replace(
            module_info[item_type]['docstring'],
            enhanced_docstring
        )

        with open(output_dir / f"{item_type}_enhanced.py", "w") as f:
            f.write(updated_code)

def main(module_name: str, object_name: str, function_name: str, output_dir: Optional[str] = None) -> None:
    """
    Main function to generate advanced documentation for a Python module.

    Args:
        module_name (str): The name of the module.
        object_name (str): The name of the object (e.g., class or function) in the module.
        function_name (str): The name of the function within the object.
        output_dir (Optional[str]): The directory to save the generated documentation. Defaults to 'docs'.

    Returns:
        None
    """
    try:
        module_info = get_module_info(module_name, object_name, function_name)
        output_path = Path(output_dir or "docs")
        generate_documentation(module_info, output_path)
        print(f"Documentation generated successfully in {output_path}")
    except (ImportError, AttributeError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main("transformers", "Trainer", "train")