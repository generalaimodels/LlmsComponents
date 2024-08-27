import subprocess
import re
from typing import Dict, Any, Tuple, Optional
from g4f.client import Client


def extract_code(content: str) -> str:
    """
    Extracts the Python code from the provided content string,
    assuming the code snippet is enclosed within triple backticks.

    Args:
        content (str): The content string containing the Python code.

    Returns:
        str: The extracted Python code or the full content if code block not found.
    """
    code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    return code_match.group(1).strip() if code_match else content.strip()


def generate_code(prompt: str) -> str:
    """
    Generates Python code based on the user's prompt using few-shot learning and chain-of-thought techniques.

    Args:
        prompt (str): The user's prompting message to generate Python code.

    Returns:
        str: The generated Python code as a string.
    """
    client = Client()  # Create a client instance

    few_shot_prompt = (
        "You are a highly skilled Python code generator. Below are examples of previous tasks and their responses:\n\n"
        "Example 1:\n"
        "Input: Write a function to calculate the factorial of a number.\n"
        "Thought Process: I need to create a recursive function that multiplies the current number by the factorial "
        "of the number minus one until it reaches 1.\n"
        "Generated Code:\n"
        "```python\n"
        "def factorial(n: int) -> int:\n"
        "    if n == 1:\n"
        "        return 1\n"
        "    else:\n"
        "        return n * factorial(n-1)\n"
        "```\n\n"
        "Example 2:\n"
        "Input: Write a function to reverse a string.\n"
        "Thought Process: The function should convert the string to a list, reverse it, and then join the list back into a string.\n"
        "Generated Code:\n"
        "```python\n"
        "def reverse_string(s: str) -> str:\n"
        "    return s[::-1]\n"
        "```\n\n"
        "Now, generate Python code for the following task:\n\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": few_shot_prompt + prompt
            }
        ],
    )

    complete_content = response.choices[0].message.content
    generated_code = extract_code(complete_content)

    return generated_code


def execute_code(code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Executes the given Python code using the subprocess module.

    Args:
        code (str): The Python code to execute.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the standard output and standard error of the execution.
    """
    try:
        # Write the code to a temporary Python file
        with open('temp_script.py', 'w') as file:
            file.write(code)

        # Use subprocess to execute the generated code
        result = subprocess.run(
            ['python3', 'temp_script.py'],
            capture_output=True,
            text=True
        )

        return result.stdout, result.stderr
    except Exception as e:
        return None, str(e)


def generate_and_execute_code(prompt: str) -> Dict[str, Any]:
    """
    Generates and executes Python code based on the user's prompt.

    Args:
        prompt (str): The user's prompting message to generate Python code.

    Returns:
        Dict[str, Any]: A dictionary containing the 'generated_code', 'output', and/or 'error' of the executed Python code.
    """
    generated_code = generate_code(prompt)

    # Attempt to append an execution if it detects a function definition
    if "def " in generated_code:
        function_name = re.search(r'def (\w+)\(', generated_code)
        if function_name:
            # Basic guess: assume the first function needs an integer or list input
            example_input = "5" if "int" in generated_code else "[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]"
            generated_code += f"\n\nprint({function_name.group(1)}({example_input}))"

    output, error = execute_code(generated_code)

    return {
        'generated_code': generated_code,
        'output': output,
        'error': error
    }


def interactive_mode():
    """
    Runs the code generator in an interactive mode, allowing the user to enter prompts
    and see the generated code and its execution results in real-time.
    """
    print("Interactive Python Code Generator. Type 'exit' to quit.")

    while True:
        user_prompt = input("Enter your prompt: ")
        if user_prompt.lower() == 'exit':
            print("Exiting the interactive mode.")
            break

        result = generate_and_execute_code(user_prompt)
        print("\nGenerated Code:\n", result['generated_code'])
        print("\nExecution Output:\n", result['output'])
        if result['error']:
            print("Execution Error:\n", result['error'])
        print("-" * 40)


# Example usage
if __name__ == '__main__':
    interactive_mode()