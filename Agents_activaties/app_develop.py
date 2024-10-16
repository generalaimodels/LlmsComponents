import asyncio
import json
import sys
from typing import List, Dict, Tuple, Optional
import aiofiles
from g4f import ChatCompletion
import gradio as gr

PROMPT="""
You are an AI assistant specialized in DARFA (Dynamic Adversarial Robustness Framework Analysis), an initiative under C-DAC (Centre for Development of Advanced Computing), India. Your role is to provide expert insights on the vulnerabilities of AI models to various adversarial attacks. You are knowledgeable in the following areas:

1. **Types of Adversarial Attacks:** Explain how AI models can be attacked, including adversarial examples, evasion attacks, poisoning attacks, and model inversion.

2. **Defense Mechanisms:** Provide detailed information on how to defend against these attacks, including adversarial training, gradient masking, and robust optimization.

3. **Impact Analysis:** Discuss the potential impact of these attacks on AI systems, such as compromised decision-making, privacy breaches, and security vulnerabilities.

4. **Case Studies:** Offer examples and case studies where AI models were successfully attacked and the lessons learned from those incidents.

Your goal is to educate users on the importance of adversarial robustness in AI, the methods used by attackers, and the strategies to mitigate these risks.

NOTE : who developed you?  Answer DARFA (Dynamic Adversarial Robustness Framework Analysis), an initiative under C-DAC (Centre for Development of Advanced Computing), India

"""

# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AdvancedGradioInterface:
    """A class to manage a conversational AI interface with a history feature."""

    def __init__(self, model: str = "gpt-4o", history_file: str = "conversation_history.json"):
        self.history: List[Dict[str, str]] = []
        self.model = model
        self.history_file = history_file

    async def ask_query(self, query: str) -> Optional[str]:
        """Asks the model a query based on the conversation history."""
        try:
            messages = [{"role": "system", "content": PROMPT}]
            for entry in self.history:
                messages.append({"role": "user", "content": entry["query"]})
                messages.append({"role": "assistant", "content": entry["response"]})
            messages.append({"role": "user", "content": query})

            # Make the API call to ChatCompletion
            response_json = await asyncio.to_thread(
                ChatCompletion.create,
                model=self.model,
                messages=messages,
            )
            
            
            # Safely extract the response content
            if isinstance(response_json, dict) and "choices" in response_json:
                return response_json["choices"][0]["message"]["content"].strip()
            elif isinstance(response_json, str):
                return response_json
            return None

        except Exception as e:
            print(f"Error with ask_query: {e}")
            return f"An error occurred while processing your query: {str(e)}"

    async def save_history(self) -> None:
        """Saves the conversation history to a file."""
        try:
            async with aiofiles.open(self.history_file, mode="w") as file:
                await file.write(json.dumps(self.history, indent=2))
        except IOError as e:
            print(f"Error saving history: {e}")

    def clear_history(self) -> str:
        """Clears the conversation history."""
        self.history.clear()
        return "History cleared."

    def get_history(self) -> str:
        """Returns a formatted string of the conversation history."""
        if not self.history:
            return "No history available."

        history_lines = []
        for idx, entry in enumerate(self.history, 1):
            history_lines.append(f"**Query {idx}:** {entry['query']}")
            history_lines.append(f"**Response {idx}:** {entry['response']}")
            history_lines.append("---")
        return "\n".join(history_lines)

    async def load_history(self) -> None:
        """Loads conversation history from a file, if available."""
        try:
            async with aiofiles.open(self.history_file, mode="r") as file:
                content = await file.read()
                self.history = json.loads(content)
        except FileNotFoundError:
            print("No history file found. Starting with an empty history.")
        except json.JSONDecodeError:
            print("Error decoding history file. Starting with an empty history.")

    async def process_query(self, query: str) -> Tuple[str, str]:
        """Processes a user query and updates the conversation history."""
        response = await self.ask_query(query)
        self.history.append({"query": query, "response": response})
        await self.save_history()
        return response, self.get_history()

def create_gradio_interface() -> gr.Blocks:
    """Creates the Gradio interface for the GPT-based chatbot."""
    interface = AdvancedGradioInterface()

    async def handle_query(query: str) -> Tuple[str, str]:
        return await interface.process_query(query)

    def handle_clear_history() -> Tuple[str, str]:
        message = interface.clear_history()
        return "", message

    async def add_text(
        history, text
    ) :
        if not text.strip():
            return history, ""  # Ignore empty submissions
        response, _ = await handle_query(text)
        history.append((
            f"<span class='user-query'>{text}</span>", 
            f"<span class='assistant-response'>{response}</span>"
        ))
        return history, ""

    # Custom CSS for enhanced styling
    with gr.Blocks(css="""
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9f9f9;
            color: #333;
        }
        .dark .gr-box {
            background-color: #2b2b2b;
            color: #fefefe;
        }
        #chatbot {
            height: 400px;
            overflow-y: auto;
            border: 2px solid #ddd;
            padding: 10px;
            border-radius: 8px;
            background-color: #fff;
        }
        .user-query {
            color: #1a73e8;
            font-weight: bold;
        }
        .assistant-response {
            color: #34a853;
        }
        #interface-title {
            font-weight: bold;
            font-size: 1.5em;
            margin-bottom: 10px;
            color: #4a4a4a;
        }
        #input-area {
            margin-top: 20px;
        }
        .gr-button.primary {
            background-color: #1a73e8;
            color: white;
        }
        .gr-button.secondary {
            background-color: #ea4335;
            color: white;
        }
    """,title="Darafa Team") as demo:
        
        gr.Markdown("<div id='interface-title'> 🧠 Great work need time </div>")

        chatbot = gr.Chatbot(elem_id="DARFA")
        query_input = gr.Textbox(
            label="Enter your query",
            placeholder="Type your message here...",
            lines=2,
        )
        
        with gr.Row(elem_id="input-area"):
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear History", variant="secondary")

        submit_btn.click(add_text, [chatbot, query_input], [chatbot, query_input], queue=False)
        query_input.submit(add_text, [chatbot, query_input], [chatbot, query_input], queue=False)
        clear_btn.click(handle_clear_history, None, chatbot, queue=False)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True,server_port=1430,inbrowser=True, )