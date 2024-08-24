import asyncio
import json
from typing import List, Dict, Any
import g4f  # Assuming this is a library for generating responses similar to OpenAI's API

# Define type aliases for clarity
Query = str
Response = Dict[str, Any]
Interaction = Dict[str, str]
Interactions = List[Interaction]

async def ask_query(query: Query) -> Response:
    """
    Asynchronously send a query to the chat model and get a response.

    Args:
        query (Query): The user query string.

    Returns:
        Response: The response from the chat model.
    """
    response = await asyncio.to_thread(
        g4f.ChatCompletion.create,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
    )
    return response# Adjust based on actual response structure

async def interact_agent(queries: List[Query]) -> Interactions:
    """
    Perform interactions by sending queries and receiving responses.

    Args:
        queries (List[Query]): A list of query strings.

    Returns:
        Interactions: A list of interactions with each query and its response.
    """
    interactions = []
    for query in queries:
        response = await ask_query(query)
        interactions.append({"query": query, "response": response})
    return interactions

def save_to_json(data: Any, filename: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (Any): The data to be saved in JSON format.
        filename (str): The name of the JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

async def main():
    """
    The main function to execute the querying and storing process.
    """
    queries = [
        "What is the capital of France?",
        "How does gravity work?",
        "Explain the theory of relativity.",
        "What are the benefits of asynchronous programming?",
        # Add more queries here, up to 20-30
    ]

    # Perform interactions between the agents
    interactions = await interact_agent(queries)

    # Save the interactions to a JSON file
    save_to_json(interactions, "interactions.json")

    # Optionally, print the results
    for interaction in interactions:
        print(f"Query: {interaction['query']}")
        print(f"Response: {interaction['response']}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main())