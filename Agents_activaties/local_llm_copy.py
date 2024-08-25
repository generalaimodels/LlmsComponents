import asyncio
import json
from typing import Dict, Any, List, Tuple
import aiohttp
import gradio as gr
import html
import markdown2

# Constants
DEFAULT_MODEL: str = "llama3.1"
DEFAULT_URL: str = "http://localhost:11434/api/chat"
DEFAULT_SEED: int = 123
DEFAULT_TEMPERATURE: float = 0.0

async def query_model(
    prompt: str,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_URL,
    seed: int = DEFAULT_SEED,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    data: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "seed": seed,
            "temperature": temperature,
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            response.raise_for_status()
            async for chunk in response.content:
                if chunk:
                    yield json.loads(chunk)["message"]["content"]

def format_conversation(conversation: List[Tuple[str, str]]) -> str:
    formatted_conversation: str = ""
    for user, bot in conversation:
        formatted_conversation += (
            f'<div class="message-container">'
            f'<div class="user-message"><span class="user-label">User:</span> {html.escape(user)}</div>'
            f'<div class="bot-message"><span class="bot-label">Bot:</span> {markdown2.markdown(bot)}</div>'
            f'</div>'
        )
    return formatted_conversation

async def chat_interface(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str]:
    try:
        result: str = ""
        async for chunk in query_model(message):
            result += chunk
            yield "", history + [(message, result)], "Processing..."
        history.append((message, result))
        yield "", history, "Query processed successfully!"
    except Exception as e:
        error_message: str = f"An error occurred: {str(e)}"
        history.append((message, error_message))
        yield "", history, "Error occurred while processing query."

def clear_conversation() -> Tuple[str, List[Tuple[str, str]], str]:
    return "", [], "Conversation cleared."

css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f0f4f8;
    color: #333;
}
.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
.chat-window {
    border: 1px solid #ccc;
    border-radius: 10px;
    padding: 15px;
    height: 400px;
    overflow-y: auto;
    background-color: #fff;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.message-container {
    margin-bottom: 15px;
}
.user-message, .bot-message {
    padding: 10px;
    border-radius: 15px;
    margin-bottom: 5px;
}
.user-message {
    background-color: #ffffa0;
}
.bot-message {
    background-color: #a0ffa0;
}
.user-label, .bot-label {
    font-weight: bold;
}
.user-label {
    color: #1565c0;
}
.bot-label {
    color: #2e7d32;
}
.status-message {
    margin-top: 10px;
    font-style: italic;
    color: #666;
}
.gr-button.primary {
    background-color: #1976d2;
    color: white;
}
.gr-button.secondary {
    background-color: #d32f2f;
    color: white;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center; color: #333;'>Advanced AI Chat Interface</h1>")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.HTML(elem_id="chat-window", label="Chat History")
            msg = gr.Textbox(label="Your message", placeholder="Type your message here...", lines=2)
            status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Conversation", variant="secondary")

    conversation_state = gr.State([])

    def update_chat(history: List[Tuple[str, str]]) -> str:
        return format_conversation(history)

    submit.click(
        chat_interface,
        inputs=[msg, conversation_state],
        outputs=[msg, conversation_state, status]
    ).then(
        update_chat,
        inputs=[conversation_state],
        outputs=[chatbot]
    )

    clear.click(
        clear_conversation,
        outputs=[msg, conversation_state, status]
    ).then(
        lambda x: "",
        outputs=[chatbot]
    )

    msg.submit(
        chat_interface,
        inputs=[msg, conversation_state],
        outputs=[msg, conversation_state, status]
    ).then(
        update_chat,
        inputs=[conversation_state],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(share=True)