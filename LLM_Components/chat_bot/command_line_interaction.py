import asyncio
import json
from typing import List, Dict, Union
import aiofiles
from colorama import Fore, init
from g4f import ChatCompletion
import sys

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set appropriate event loop policy for Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AdvancedCLI:
    def __init__(self, model: str = "gpt-3.5-turbo", history_file: str = "conversation_history.json"):
        self.history: List[Dict[str, str]] = []
        self.model: str = model
        self.history_file: str = history_file

    async def ask_query(self, query: str) -> str:
        """
        Asynchronously send a query to the GPT model and return the response.
        """
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for entry in self.history:
                messages.append({"role": "user", "content": entry["query"]})
                messages.append({"role": "assistant", "content": entry["response"]})
            messages.append({"role": "user", "content": query})

            response = await asyncio.to_thread(
                ChatCompletion.create,
                model=self.model,
                messages=messages,
            )
            return response
        except Exception as e:
            print(f"{Fore.RED}Error occurred while processing query: {e}")
            return "An error occurred while processing your query."

    async def save_history(self) -> None:
        """
        Asynchronously save the conversation history to a JSON file.
        """
        try:
            async with aiofiles.open(self.history_file, mode='w') as f:
                await f.write(json.dumps(self.history, indent=2))
            print(f"{Fore.YELLOW}History saved successfully.")
        except IOError as e:
            print(f"{Fore.RED}Error saving history: {e}")

    def clear_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.history.clear()
        print(f"{Fore.YELLOW}History cleared.")

    def print_history(self) -> None:
        """
        Print the conversation history.
        """
        if not self.history:
            print(f"{Fore.YELLOW}No history available.")
            return

        for idx, entry in enumerate(self.history, 1):
            print(f"{Fore.CYAN}Query {idx}: {entry['query']}")
            print(f"{Fore.GREEN}Response {idx}: {entry['response']}")
            print("---")

    async def load_history(self) -> None:
        """
        Asynchronously load the conversation history from a JSON file.
        """
        try:
            async with aiofiles.open(self.history_file, mode='r') as f:
                content = await f.read()
                self.history = json.loads(content)
            print(f"{Fore.YELLOW}History loaded successfully.")
        except FileNotFoundError:
            print(f"{Fore.YELLOW}No history file found. Starting with empty history.")
        except json.JSONDecodeError:
            print(f"{Fore.RED}Error decoding history file. Starting with empty history.")

    async def interactive_mode(self) -> None:
        """
        Run the CLI in interactive mode.
        """
        print(f"{Fore.MAGENTA}Welcome to the Advanced CLI!")
        print("Enter your query, or type 'history' to view history, 'clear' to clear history, or 'quit' to exit.")

        await self.load_history()

        while True:
            user_input = input(f"{Fore.YELLOW}\nEnter your query: ").strip().lower()

            if user_input in['quit',"q","exit","bye","good bye","close","Good Bye"]:
                break
            elif user_input == 'history':
                self.print_history()
            elif user_input == 'clear':
                self.clear_history()
            else:
                response = await self.ask_query(user_input)
                print(f"{Fore.GREEN}Response: {response}")

                self.history.append({
                    "query": user_input,
                    "response": response
                })
                await self.save_history()

        print(f"{Fore.MAGENTA}Thank you for using the Advanced CLI. Goodbye!")

async def main() -> None:
    cli = AdvancedCLI(
        history_file="conversation_history.json"
    )
    await cli.interactive_mode()

# if __name__ == "__main__":
#     asyncio.run(main())