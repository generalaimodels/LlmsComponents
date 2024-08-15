import asyncio
import g4f

async def ask_query(query):
    response = await asyncio.to_thread(g4f.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
    )
    return response

async def main():
    queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Explain the theory of relativity.",
        # Add more queries here, up to 20-30
    ]

    tasks = []
    for query in queries:
        task = asyncio.create_task(ask_query(query))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for query, result in zip(queries, results):
        print(f"Query: {query}")
        print(f"Response: {result}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main())