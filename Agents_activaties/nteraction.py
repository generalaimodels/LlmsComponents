import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path
import g4f

async def ask_query(query: str) -> str:
    """
    Asynchronously ask a query to the LLM.

    Args:
        query (str): The input query.

    Returns:
        str: The response from the LLM.
    """
    response = await asyncio.to_thread(
        g4f.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
    )
    return response

async def process_query(query: str) -> Dict[str, str]:
    """
    Process a single query and get the response.

    Args:
        query (str): The input query.

    Returns:
        Dict[str, str]: A dictionary containing the query and response.
    """
    response = await ask_query(query)
    return {"query": query, "response": response}

def save_to_json(data: List[Dict[str, str]], filename: str = "query_responses.json") -> None:
    """
    Save the query and response data to a JSON file.

    Args:
        data (List[Dict[str, str]]): The list of query-response pairs.
        filename (str, optional): The name of the JSON file. Defaults to "query_responses.json".
    """
    file_path = Path(filename)
    mode = "a" if file_path.exists() else "w"
    
    with open(file_path, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write(",\n")
        json.dump(data, f, indent=2, ensure_ascii=False)

async def main() -> None:
    """
    Main function to handle user input, process queries, and save results.
    """
    queries: List[str] = []
    
    print("Enter your queries (press Enter twice to finish):")
    while True:
        query = input().strip()
        if not query:
            break
        queries.append(query)

    tasks = [asyncio.create_task(process_query(query)) for query in queries]
    results = await asyncio.gather(*tasks)

    save_to_json(results)
    print(f"Queries and responses have been saved to query_responses.json")

if __name__ == "__main__":
    asyncio.run(main())