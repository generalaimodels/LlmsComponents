import asyncio
import json
from typing import Dict, Any, List, Tuple
import aiohttp
import gradio as gr
import html

# Constants
DEFAULT_MODEL = "llama3.1"
DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_SEED = 123
DEFAULT_TEMPERATURE = 0.0

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
            return await process_streaming_response(response)

async def process_streaming_response(response: aiohttp.ClientResponse) -> str:
    response_data = ""
    async for line in response.content:
        if line:
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data

def format_conversation(conversation: List[Tuple[str, str]]) -> str:
    formatted_conversation = ""
    for i, (user, bot) in enumerate(conversation):
        formatted_conversation += (
            f'<div style="margin-bottom: 10px;">'
            f'<span style="color: #2874A6; font-weight: bold;">User:</span> {html.escape(user)}<br>'
            f'<span style="color: #229954; font-weight: bold;">Bot:</span> {html.escape(bot)}'
            f'</div>'
        )
    return formatted_conversation

def chat_interface(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    try:
        result = asyncio.run(query_model(message))
        history.append((message, result))
        return "", history
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        history.append((message, error_message))
        return "", history

def clear_conversation() -> Tuple[str, List[Tuple[str, str]]]:
    return "", []

css = """
.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
.chat-window {
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
    height: 400px;
    overflow-y: auto;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center;'>Advanced Chat Interface</h1>")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.HTML(label="Chat History")
            msg = gr.Textbox(label="Your message", placeholder="Type your message here...")
            with gr.Row():
                submit = gr.Button("Send")
                clear = gr.Button("Clear Conversation")

    conversation_state = gr.State([])

    def update_chat(history: List[Tuple[str, str]]) -> str:
        return format_conversation(history)

    submit.click(
        chat_interface,
        inputs=[msg, conversation_state],
        outputs=[msg, conversation_state]
    ).then(
        update_chat,
        inputs=[conversation_state],
        outputs=[chatbot]
    )

    clear.click(
        clear_conversation,
        outputs=[msg, conversation_state]
    ).then(
        lambda x: "",
        outputs=[chatbot]
    )

    msg.submit(
        chat_interface,
        inputs=[msg, conversation_state],
        outputs=[msg, conversation_state]
    ).then(
        update_chat,
        inputs=[conversation_state],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(share=True,server_port=8080)