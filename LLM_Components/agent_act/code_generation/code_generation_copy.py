import subprocess
import re
import json
import os
from typing import Dict, Any, Tuple, Optional, List
from g4f.client import Client
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeGenerationSystem:
    def __init__(self, max_correction_attempts: int = 3, default_timeout: int = 30, max_history_size: int = 100):
        self.client = Client()
        self.history: List[Dict[str, Any]] = []
        self.max_correction_attempts = max_correction_attempts
        self.default_timeout = default_timeout
        self.max_history_size = max_history_size

    def extract_code(self, content: str) -> str:
        """Extract Python code from the provided content string."""
        code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
        return code_match.group(1).strip() if code_match else content.strip()

    def generate_code(self, prompt: str) -> str:
        """Generate Python code based on the user's prompt using the GPT model."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python code generator. Generate code based on the user's prompt."},
                {"role": "user", "content": prompt}
            ],
        )
        complete_content = response.choices[0].message.content
        generated_code = self.extract_code(complete_content)
        logger.info(f"Generated Code:\n{generated_code}")
        return generated_code

    def execute_code(self, code: str, timeout: int = None) -> Tuple[Optional[str], Optional[str]]:
        """Execute the given Python code using the subprocess module."""
        if timeout is None:
            timeout = self.default_timeout

        try:
            with open('temp_script.py', 'w',encoding="UTF-8") as file:
                file.write(code)

            result = subprocess.run(
                ['python3', 'temp_script.py'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return None, f"Execution timed out after {timeout} seconds"
        except Exception as e:
            return None, str(e)

    def generate_and_execute_code(self, prompt: str, timeout: int = None) -> Dict[str, Any]:
        """Generate and execute Python code based on the user's prompt."""
        generated_code = self.generate_code(prompt)
        output, error = self.execute_code(generated_code, timeout)

        execution_result = {
            'prompt': prompt,
            'code': generated_code,
            'output': output,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.add_to_history(execution_result)
        
        attempts = 0
        while error and attempts < self.max_correction_attempts:
            attempts += 1
            logger.info(f"Attempt {attempts} to correct the code")
            corrected_code = self.correct_code(prompt, generated_code, error)
            if corrected_code == generated_code:
                logger.info("No changes made in the correction attempt")
                break
            output, error = self.execute_code(corrected_code, timeout)
            execution_result = {
                'prompt': prompt,
                'code': corrected_code,
                'output': output,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            self.add_to_history(execution_result)
            generated_code = corrected_code

        return execution_result

    def correct_code(self, prompt: str, code: str, error: str) -> str:
        """Attempt to correct the code based on the error message."""
        correction_prompt = f"""
        The following code was generated for the prompt: "{prompt}"
        
        Code:
        {code}
        
        This code produced the following error:
        {error}
        
        Please correct the code to fix this error and optimize for performance.
        If you cannot fix the error, return the original code.
        """
        return self.generate_code(correction_prompt)

    def add_to_history(self, execution_result: Dict[str, Any]):
        """Add an execution result to the history, maintaining the maximum history size."""
        self.history.append(execution_result)
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]

    def save_history(self, filename: str = 'code_generation_history.json'):
        """Save the code generation history to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"History saved to {filename}")

    def load_history(self, filename: str = 'code_generation_history.json'):
        """Load the code generation history from a JSON file."""
        try:
            with open(filename, 'r') as f:
                self.history = json.load(f)
            logger.info(f"History loaded from {filename}")
            if len(self.history) > self.max_history_size:
                self.history = self.history[-self.max_history_size:]
                logger.info(f"History truncated to {self.max_history_size} entries")
        except FileNotFoundError:
            logger.warning(f"History file {filename} not found. Starting with an empty history.")

def interactive_loop():
    system = CodeGenerationSystem()
    system.load_history()

    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        
        timeout = 120

        result = system.generate_and_execute_code(prompt, timeout)
        
        print("\nExecution Result:")
        print("Output:", result['output'])
        if result['error']:
            print("Error:", result['error'])
        
        print("\nGenerated Code:")
        print(result['code'])

        system.save_history()

if __name__ == '__main__':
    interactive_loop()