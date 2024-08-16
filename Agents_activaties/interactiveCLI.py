import asyncio
import json
from typing import List, Dict, Union
import g4f
import colorama
import sys
# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

# Set appropriate event loop policy for Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AdvancedCLI:
    def __init__(self):
        self.history: List[Dict[str, Union[str, List[str]]]] = []

    async def ask_query(self, query: str) -> str:
        """
        Asynchronously send a query to the GPT model and return the response.
        """
        response = await asyncio.to_thread(
            g4f.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
        )
        return response

    def save_history(self) -> None:
        """
        Save the conversation history to a JSON file.
        """
        with open("conversation_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def clear_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.history.clear()
        print(colorama.Fore.YELLOW + "History cleared.")

    def print_history(self) -> None:
        """
        Print the conversation history.
        """
        for idx, entry in enumerate(self.history, 1):
            print(f"{colorama.Fore.CYAN}Query {idx}: {entry['query']}")
            print(f"{colorama.Fore.GREEN}Response {idx}: {entry['response']}")
            print("---")

    async def interactive_mode(self) -> None:
        """
        Run the CLI in interactive mode.
        """
        print(colorama.Fore.MAGENTA + "Welcome to the Advanced CLI!")
        print("Enter your query, or type 'history' to view history, 'clear' to clear history, or 'quit' to exit.")

        while True:
            user_input = input(colorama.Fore.YELLOW + "\nEnter your query: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'history':
                self.print_history()
            elif user_input.lower() == 'clear':
                self.clear_history()
            else:
                response = await self.ask_query(user_input)
                print(f"{colorama.Fore.GREEN}Response: {response}")

                self.history.append({
                    "query": user_input,
                    "response": response
                })

        self.save_history()
        print(colorama.Fore.MAGENTA + "Thank you for using the Advanced CLI. Goodbye!")

async def main() -> None:
    cli = AdvancedCLI()
    await cli.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())