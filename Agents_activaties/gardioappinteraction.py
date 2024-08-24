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
        self.history: List[Dict[str, str]] = []
        self.model: str = model
        self.history_file: str = history_file

    async def ask_query(self, query: str) -> str:
        try:
            messages = [{"role": "system", "content": "You are a helpful  Dynamic Adversarial Robustness Analysis Framework for AI assistant. developed by cdacb"}]
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
            return f"An error occurred while processing your query: {str(e)}"

    async def save_history(self) -> None:
        try:
            async with aiofiles.open(self.history_file, mode='w') as f:
                await f.write(json.dumps(self.history, indent=2))
        except IOError as e:
            print(f"Error saving history: {e}")

    def clear_history(self) -> str:
        self.history.clear()
        return "History cleared."

    def get_history(self) -> str:
        if not self.history:
            return "No history available."

        history_str = ""
        for idx, entry in enumerate(self.history, 1):
            history_str += f"Query {idx}: {entry['query']}\n"
            history_str += f"Response {idx}: {entry['response']}\n"
            history_str += "---\n"
        return history_str

    async def load_history(self) -> None:
        try:
            async with aiofiles.open(self.history_file, mode='r') as f:
                content = await f.read()
                self.history = json.loads(content)
        except FileNotFoundError:
            print("No history file found. Starting with empty history.")
        except json.JSONDecodeError:
            print("Error decoding history file. Starting with empty history.")

    async def process_query(self, query: str) -> Tuple[str, str]:
        response = await self.ask_query(query)
        self.history.append({
            "query": query,
            "response": response
        })
        await self.save_history()
        return response, self.get_history()

def create_gradio_interface() -> gr.Interface:
    interface = AdvancedGradioInterface()

    async def handle_query(query: str) -> Tuple[str, str]:
        return await interface.process_query(query)

    def handle_clear_history() -> Tuple[str, str]:
        message = interface.clear_history()
        return "", message

    with gr.Blocks(css="#chatbot {height: 350px; overflow: auto;}") as demo:
        gr.Markdown("#  🤖  🤖 Dynamic Adversarial Robustness Analysis Framework for AI 🤖  🤖 ")
        
        chatbot = gr.Chatbot(elem_id="chatbot")
        query_input = gr.Textbox(label="Enter your query", placeholder="Type your message here...")
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear History", variant="secondary")

        async def add_text(history, text):
            if text.strip() == "":
                return history
            response, _ = await handle_query(text)
            history.append((text, response))
            return history, ""

        submit_btn.click(add_text, [chatbot, query_input], [chatbot, query_input], queue=False)
        query_input.submit(add_text, [chatbot, query_input], [chatbot, query_input], queue=False)
        clear_btn.click(lambda: None, None, chatbot, queue=False)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)