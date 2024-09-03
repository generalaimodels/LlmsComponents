import importlib
import inspect
import re
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path

from g4f.client import Client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE = "api_search_results.txt"


class APISearcher:
    def __init__(self):
        self.client = Client()
        self.loaded_modules: Dict[str, Any] = {}

    def load_module(self, module_name: str) -> Optional[Any]:
        """Load a module dynamically."""
        if module_name not in self.loaded_modules:
            try:
                self.loaded_modules[module_name] = importlib.import_module(module_name)
                logging.info(f"Successfully imported module '{module_name}'")
            except ImportError:
                logging.error(f"Unable to import module '{module_name}'")
                return None
        return self.loaded_modules[module_name]

    def get_module_functions(self, module: Any) -> List[Tuple[str, Any]]:
        """Get all functions from a module."""
        return inspect.getmembers(module, lambda x: inspect.isfunction(x) or inspect.isclass(x))

    def parse_gpt_response(self, response_content: str) -> List[Tuple[str, str]]:
        """Parse the GPT response and extract module and function suggestions."""
        suggestions = []
        lines = response_content.split('\n')
        for line in lines:
            match = re.match(r'(\S+)\.(\S+)', line)
            if match:
                module_name = match.group(1).strip()
                function_name = match.group(2).strip()
                suggestions.append((module_name, function_name))
        return suggestions

    def search_api(self, prompt: str) -> List[Tuple[str, Any]]:
        """Search for the best possible API based on the given prompt."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Python expert assistant."},
                {"role": "user", "content": f"Based on the prompt '{prompt}', suggest potential functions, classes, or methods from any Python module that might be relevant. Format your response as a list of 'module_name.function_or_class' items, one per line."},
            ],
        )

        # Parse the response to get module and function suggestions
        suggestions = self.parse_gpt_response(response.choices[0].message.content)

        results: List[Tuple[str, Any]] = []
        for module_name, function_name in suggestions:
            module = self.load_module(module_name)
            if module:
                attr = module
                for part in function_name.split('.'):
                    if hasattr(attr, part):
                        attr = getattr(attr, part)
                    else:
                        attr = None
                        logging.warning(f"'{function_name}' not found in the module '{module_name}'.")
                        break
                if attr:
                    results.append((f"{module_name}.{function_name}", attr))
        return results

    def save_results_to_file(self, results: List[Tuple[str, Any]], file_path: Path):
        """Save the search results to a text file."""
        if not results:
            message = "No matching APIs found."
            logging.info(message)
        else:
            with file_path.open('w') as f:
                f.write("Matching APIs:\n")
                for api_name, func in results:
                    f.write(f"- {api_name}\n")
                    if inspect.isclass(func):
                        f.write(f"  Description: {func.__doc__ or 'No description available'}\n")
                        f.write("  Methods:\n")
                        for method_name, method in inspect.getmembers(func, inspect.isfunction):
                            if not method_name.startswith('_'):
                                f.write(f"    - {method_name}: {method.__doc__ or 'No description available'}\n")
                    else:
                        f.write(f"  Description: {func.__doc__ or 'No description available'}\n")
                    f.write("\n")
            logging.info(f"Results written to {file_path}")

    def run(self):
        """Main loop to interact with the user."""
        try:
            while True:
                prompt = input("Enter your API search prompt (or 'q' to quit): ")
                if prompt.lower() == 'q':
                    break
                results = self.search_api(prompt)
                file_path = Path(LOG_FILE)
                self.save_results_to_file(results, file_path)
                print(f"Results have been saved to {file_path}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")


# if __name__ == "__main__":
#     searcher = APISearcher()
#     searcher.run()