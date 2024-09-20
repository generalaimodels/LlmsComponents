# client_interface.py
import asyncio
import sys
from g4f.client import Client
from ragsystemspeed import setup_logger

logger = setup_logger()

# Set the appropriate event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



def generate_prompt(query, context, use_chain_of_thoughts=True, use_few_shot=False, use_tree_of_thoughts=False, use_complex_splitting=True):
    """
    Generates a structured prompt for a Retrieval-Augmented Generation (RAG) system using advanced concepts.
    
    Args:
        query (str): User input query.
        context (str): Context or background information provided by the user.
        use_chain_of_thoughts (bool): Whether to apply the chain of thoughts methodology.
        use_few_shot (bool): Whether to include few-shot learning examples.
        use_tree_of_thoughts (bool): Whether to apply tree of thoughts reasoning.
        use_complex_splitting (bool): Whether to break complex queries into smaller tasks.

    Returns:
        str: A structured prompt.
    """

    # Base template
    prompt = f"""
User Query: {query}
Context: {context}
"""

    # Chain of Thoughts
    if use_chain_of_thoughts:
        prompt += "\nChain of Thoughts:\n"
        prompt += "1. Break down the query into smaller questions.\n"
        prompt += "2. Generate reasoning steps to solve the query iteratively.\n"

    # Few-Shot Learning
    if use_few_shot:
        prompt += "\nFew-Shot Learning:\n"
        prompt += "1. Retrieve and include relevant examples or similar past cases.\n"
        prompt += "2. Use these examples to guide the response generation.\n"

    # Tree of Thoughts
    if use_tree_of_thoughts:
        prompt += "\nTree of Thoughts:\n"
        prompt += "1. Explore multiple interpretations or branches of the query.\n"
        prompt += "2. Retrieve relevant answers for each interpretation.\n"
        prompt += "3. Weigh and combine different solutions into a final response.\n"

    # Complex to Basic Splitting
    if use_complex_splitting:
        prompt += "\nComplex to Basic Splitting:\n"
        prompt += "1. Break the complex query into smaller, manageable tasks.\n"
        prompt += "2. Retrieve information for each sub-task.\n"
        prompt += "3. Aggregate the results to generate a comprehensive response.\n"

    # Final Output
    prompt += "\nFinal Output:\n"
    prompt += "Generate a coherent, detailed response based on the above steps, synthesizing the retrieved knowledge."

    return prompt

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