import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
from colorama import Fore, Style, init
import asyncio

# Initialize colorama
init(autoreset=True)

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class CLIHistory:
    def __init__(self, history_dir: str = "./history"):
        self.history_dir = Path(history_dir)
        self.history_file = self.history_dir / "cli_interaction_history.json"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_history_file()

    def _initialize_history_file(self):
        if not self.history_file.exists():
            with open(self.history_file, 'w') as f:
                json.dump([], f, indent=4)

    def save_interaction(self, user_input: str, response: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        interaction = {
            "timestamp": timestamp,
            "query": user_input,
            "response": response
        }

        with open(self.history_file, 'r+') as f:
            history = json.load(f)
            history.append(interaction)
            f.seek(0)
            json.dump(history, f, indent=4)

    def load_history(self) -> List[Dict[str, Any]]:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def display_history(self):
        history = self.load_history()
        for entry in history:
            print(f"{Fore.YELLOW}{entry['timestamp']} - Query: {entry['query']}")
            print(f"{Fore.GREEN}Response: {entry['response']}\n")


class CommandLineInterface:
    def __init__(self, client: Any, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.history = CLIHistory()

    def color_print(self, message: str, color: str):
        color_map = {
            "yellow": Fore.YELLOW,
            "green": Fore.GREEN,
            "red": Fore.RED
        }
        print(color_map.get(color, Fore.RESET) + message + Style.RESET_ALL)
    
    def get_response_stream(self, query: str) -> str:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                stream=False,
            )
            response = ""
            line_buffer = ""
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        line_buffer += content
                        if '\n' in line_buffer:
                            lines = line_buffer.split('\n')
                            for line in lines[:-1]:  # Print all except the last unfinished line
                                self.color_print(line, "green")
                            line_buffer = lines[-1]  # Keep the last unfinished line in the buffer
            if line_buffer:
                self.color_print(line_buffer, "green")  # Print any remaining content in buffer
            return response
        except Exception as e:
            self.color_print(f"Error: {str(e)}", "red")
            return ""

    def prompt_user(self, custom_prompt: Optional[str] = None):
        prompt_text = custom_prompt if custom_prompt else "You: "
        try:
            user_input = input(Fore.YELLOW + prompt_text + Style.RESET_ALL).strip()
            response = self.get_response_stream(user_input)
            self.history.save_interaction(user_input, response)
        except (KeyboardInterrupt, EOFError):
            self.color_print("\nExiting the application.", "red")
            sys.exit(0)

    def run(self, custom_prompt: Optional[str] = None):
        while True:
            self.prompt_user(custom_prompt)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced CLI Interface with AI interaction.")
    parser.add_argument("--prompt", type=str, help="Customize the CLI prompt.")
    parser.add_argument("--history", action="store_true", help="Display the interaction history.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Initialize the client, make sure that 'g4f.client' is properly configured
    from g4f.client import Client
    client = Client()

    cli = CommandLineInterface(client)

    if args.history:
        cli.history.display_history()
    else:
        cli.run(args.prompt)


if __name__ == "__main__":
    main()