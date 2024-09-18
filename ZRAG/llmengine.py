# client_interface.py
import asyncio
import sys
from g4f.client import Client
from raglogger import setup_logger

logger = setup_logger()

# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_llm_response(model: str, user_input: str) -> str:
    """
    Sends the user input to the LLM and retrieves the model's response.
    :param model: LLM model identifier.
    :param user_input: User's query.
    :return: The model's textual response.
    """
    try:
        client = Client()
        logger.info(f"Sending user input to LLM model: {model}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_input}],
        )
        
        message = response.choices[0].message.content
        logger.info("Received response from LLM.")
        return message
    
    except Exception as e:
        logger.error(f"Error during LLM interaction: {str(e)}")
        return "Error occurred while retrieving response from LLM."