import importlib
import inspect
import subprocess
import re
import sys
from typing import Dict, Any, Tuple, Optional, List

from g4f.client import Client

def extract_code(content: str) -> str:
    """
    Extract Python code from the provided content string.

    Args:
        content (str): The content string containing the Python code.

    Returns:
        str: The extracted Python code.
    """
    code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    return code_match.group(1).strip() if code_match else content.strip()

def get_available_modules() -> List[str]:
    """
    Get a list of available modules in the current environment.

    Returns:
        List[str]: A list of available module names.
    """
    return list(sys.modules.keys())

def get_module_apis(module_name: str) -> Dict[str, Any]:
    """
    Get the APIs (functions and classes) of a given module.

    Args:
        module_name (str): The name of the module.

    Returns:
        Dict[str, Any]: A dictionary of API names and their corresponding objects.
    """
    try:
        module = importlib.import_module(module_name)
        return {
            name: obj for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) or inspect.isclass(obj)
        }
    except ImportError:
        return {}

def generate_code(prompt: str, available_modules: List[str]) -> str:
    """
    Generate Python code based on the user's prompt and available modules.

    Args:
        prompt (str): The user's prompting message to generate Python code.
        available_modules (List[str]): List of available modules in the environment.

    Returns:
        str: The generated Python code as a string.
    """
    client = Client()

    # Prepare the context with available modules and their APIs
    context = "Available modules:\n"
    for module in available_modules[:10]:  # Limit to first 10 modules to avoid overloading
        context += f"- {module}\n"
        apis = get_module_apis(module)
        for api_name, api_obj in list(apis.items())[:5]:  # Limit to first 5 APIs per module
            context += f"  - {api_name}\n"

    # Create a chat completion request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python code generator. Generate code based on the user's prompt "
                    "and the available modules and APIs in the current environment. "
                    "Include a test case that demonstrates the usage of the generated code "
                    "and prints the result."
                )
            },
            {
                "role": "user",
                "content": f"{context}\n\nUser prompt: {prompt}"
            }
        ],
    )

    return extract_code(response.choices[0].message.content)

def execute_code(code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Execute the given Python code using the subprocess module.

    Args:
        code (str): The Python code to execute.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the standard output and standard error of the execution.
    """
    try:
        with open('temp_script.py', 'w') as file:
            file.write(code)

        result = subprocess.run(
            [sys.executable, 'temp_script.py'],
            capture_output=True,
            text=True,
            timeout=30  # Set a timeout to prevent infinite loops
        )

        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return None, "Execution timed out"
    except Exception as e:
        return None, str(e)

def generate_and_execute_code(prompt: str) -> Dict[str, Any]:
    """
    Generate and execute Python code based on the user's prompt.

    Args:
        prompt (str): The user's prompting message to generate Python code.

    Returns:
        Dict[str, Any]: A dictionary containing the 'code', 'output', and/or 'error' of the executed Python code.
    """
    available_modules = get_available_modules()
    generated_code = generate_code(prompt, available_modules)
    output, error = execute_code(generated_code)

    return {
        'code': generated_code,
        'output': output,
        'error': error
    }

prompt_message = "Create a function that calculates the factorial of a number and test it with the number 5"
result = generate_and_execute_code(prompt_message)
print("Generated Code:\n", result['code'])
print("\nExecution Output:\n", result['output'])
if result['error']:
    print("\nExecution Error:\n", result['error'])