import asyncio
import json
import sys
from typing import List, Dict, Tuple
import aiofiles
from g4f import ChatCompletion
import gradio as gr

# Set appropriate event loop policy for Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AdvancedGradioInterface:
    def __init__(self, model: str = "gpt-4o-mini", history_file: str = "conversation_history.json"):
        self.history: List[Tuple[str, str]] = []  # List of (user_message, assistant_response) tuples
        self.model: str = model
        self.history_file: str = history_file

    async def ask_query(self, query: str) -> str:
        """Ask a query to the AI model and return the response."""
        try:
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            for entry in self.history:
                messages.append({"role": "user", "content": entry[0]})
                messages.append({"role": "assistant", "content": entry[1]})
            messages.append({"role": "user", "content": query})

            # Fetch the response from the API
            response = await asyncio.to_thread(ChatCompletion.create, model=self.model, messages=messages)

            if isinstance(response, str):
                return response
            else:
                return "Error: Unexpected response format."

        except Exception as e:
            return f"An error occurred: {str(e)}"

    async def save_history(self) -> None:
        """Save the conversation history to a JSON file."""
        try:
            async with aiofiles.open(self.history_file, 'w') as f:
                await f.write(json.dumps(self.history, indent=2))
        except IOError as e:
            print(f"Error saving history: {e}")

    def clear_history(self) -> None:
        """Clear the conversation history and save the empty state."""
        self.history.clear()
        asyncio.run(self.save_history())

    def get_history(self) -> List[Tuple[str, str]]:
        """Return the conversation history."""
        return self.history

    async def load_history(self) -> None:
        """Load the conversation history from a file."""
        try:
            async with aiofiles.open(self.history_file, 'r') as f:
                content = await f.read()
                self.history = json.loads(content)
        except FileNotFoundError:
            print("No history file found. Starting with an empty history.")
        except json.JSONDecodeError:
            print("Error decoding history file. Starting with an empty history.")

    async def process_query(self, query: str) -> List[Tuple[str, str]]:
        """Process a user query and update the conversation history."""
        response = await self.ask_query(query)
        self.history.append((query, response))
        await self.save_history()
        return self.get_history()


def create_gradio_interface() -> gr.Interface:
    """Create and return the Gradio interface."""
    interface = AdvancedGradioInterface()

    async def handle_query(query: str) -> List[Tuple[str, str]]:
        return await interface.process_query(query)

    def handle_clear_history() -> List[Tuple[str, str]]:
        interface.clear_history()
        return []

    with gr.Blocks(css="#chatbot {height: 350px; overflow: auto;}") as demo:
        gr.Markdown("# 🤖 Dynamic Adversarial Robustness Analysis Framework for AI 🤖")

        chatbot = gr.Chatbot(elem_id="chatbot")
        query_input = gr.Textbox(label="Enter your query", placeholder="Type your message here...")

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear History", variant="secondary")

        async def add_text(history: List[Tuple[str, str]], text: str) -> Tuple[List[Tuple[str, str]], str]:
            if text.strip() == "":
                return history, ""

            updated_history = await handle_query(text)
            return updated_history, ""

        submit_btn.click(add_text, [chatbot, query_input], [chatbot, query_input], queue=False)
        query_input.submit(add_text, [chatbot, query_input], [chatbot, query_input], queue=False)
        clear_btn.click(handle_clear_history, [], [chatbot], queue=False)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)