import subprocess
import re
import os
import json
import time
from typing import Dict, Any, Tuple, Optional
from g4f.client import Client

# Directory for storing code versions and execution history
HISTORY_DIR = 'code_history'
os.makedirs(HISTORY_DIR, exist_ok=True)

MAX_RETRIES = 3  # Maximum number of retries for error correction
DEFAULT_TIMEOUT = 30  # Default execution timeout in seconds

def extract_code(content: str) -> str:
    code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return content.strip()

def save_code_version(code: str, version: int) -> str:
    file_path = os.path.join(HISTORY_DIR, f'code_v{version}.py')
    with open(file_path, 'w') as file:
        file.write(code)
    return file_path

def save_execution_result(version: int, output: Optional[str], error: Optional[str]) -> None:
    result = {
        'version': version,
        'output': output,
        'error': error
    }
    file_path = os.path.join(HISTORY_DIR, f'result_v{version}.json')
    with open(file_path, 'w') as file:
        json.dump(result, file, indent=4)

def generate_code(prompt: str) -> str:
    client = Client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a Python code generator. Generate code based on the user's prompt."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    complete_content = response.choices[0].message.content
    generated_code = extract_code(complete_content)
    print("Generated Code:\n", generated_code)
    
    return generated_code

def execute_code(code: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[str], Optional[str]]:
    try:
        start_time = time.time()
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        duration = time.time() - start_time
        print(f"Execution completed in {duration:.2f} seconds")
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return None, "Execution timed out"
    except Exception as e:
        return None, str(e)

def generate_and_execute_code(prompt: str) -> Dict[str, Any]:
    version = len([name for name in os.listdir(HISTORY_DIR) if name.startswith('code_v')]) + 1
    retries = 0

    while retries < MAX_RETRIES:
        generated_code = generate_code(prompt)
        code_file_path = save_code_version(generated_code, version)
        output, error = execute_code(generated_code)

        save_execution_result(version, output, error)

        if not error:
            break

        print(f"Retry {retries + 1}/{MAX_RETRIES}: Encountered an error, trying to correct...")
        retries += 1

    return {
        'version': version,
        'code_file': code_file_path,
        'output': output,
        'error': error
    }

# Example usage
if __name__ == '__main__':
    prompt_message = "Write a function to calculate the factorial of a number"
    result = generate_and_execute_code(prompt_message)
    if result['output']:
        print("Execution Output:\n", result['output'])
    if result['error']:
        print("Execution Error:\n", result['error'])
    print(f"Code Version: {result['version']} saved at {result['code_file']}")