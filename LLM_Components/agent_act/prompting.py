from typing import List, Optional
from g4f.client import Client
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
# Setting the event loop policy for Windows platforms if necessary
import sys
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



@lru_cache(maxsize=128)
def get_client() -> Client:
    """
    Get or create a Client instance.
    
    Returns:
        Client: An instance of the g4f Client.
    """
    return Client()

def process_model_output(output: str) -> List[str]:
    """
    Process the model's output to extract fine-tuned prompts.
    
    Args:
        output (str): The raw output from the model.
    
    Returns:
        List[str]: A list of processed fine-tuned prompts.
    """
    # Implement more sophisticated processing logic here
    return [prompt.strip() for prompt in output.split('\n') if prompt.strip()]

def generate_single_prompt(client: Client, task: str) -> Optional[str]:
    """
    Generate a single fine-tuned prompt for a given task.
    
    Args:
        client (Client): The g4f Client instance.
        task (str): The task to generate a prompt for.
    
    Returns:
        Optional[str]: The generated prompt, or None if generation failed.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Generate a fine-tuned prompt for this task: {task}"}],
    )
    return response.choices[0].message.content.strip()
  

def generate_fine_tuned_prompts(user_query: str, max_workers: int = 5) -> List[str]:
    """
    Break down the user's query into smaller tasks and generate fine-tuned prompts
    for each subtask using Few-Shot and Chain of Thought prompting techniques.
    
    Args:
        user_query (str): The user's query.
        max_workers (int): Maximum number of concurrent workers for prompt generation.
    
    Returns:
        List[str]: A list of fine-tuned prompts for each subtask.
    
    Raises:
        PromptGenerationError: If no prompts could be generated.
    """
    client = get_client()
    
    
    # Initial breakdown of the query into tasks
    initial_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Break down this query into smaller tasks: {user_query}"}
        ],
    )
    
    
    tasks = process_model_output(initial_response.choices[0].message.content)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(generate_single_prompt, client, task): task for task in tasks}
        prompts = []
        
        for future in as_completed(future_to_task):
            prompt = future.result()
            if prompt:
                prompts.append(prompt)
    return prompts
    
   

if __name__ == "__main__":
    user_query = "Explain the process of photosynthesis in plants"
    
    fine_tuned_prompts = generate_fine_tuned_prompts(user_query)
    for i, prompt in enumerate(fine_tuned_prompts, 1):
        print(f"Prompt {i}: {prompt}")
