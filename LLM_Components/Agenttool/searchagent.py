from typing import List, Dict, Any, Optional
import logging
from g4f.client import Client
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchAgent:
    def __init__(self, gpt_model: str = "gpt-4o-mini", max_search_results: int = 7):
        self.client = Client()
        self.gpt_model = gpt_model
        self.max_search_results = max_search_results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _gpt_query(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Query GPT model with retry mechanism."""
        try:
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying GPT: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search with retry mechanism."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_search_results))
            return results
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            raise

    def chain_of_thought_prompt(self, query: str) -> str:
        """Generate a chain of thought prompt for the given query."""
        prompt = (
            f"Given the query: '{query}', let's approach this step-by-step:\n"
            "1. Identify the main topic and any subtopics.\n"
            "2. Consider what specific information is being asked for.\n"
            "3. Think about potential sources of information for this query.\n"
            "4. Formulate a plan to find and synthesize the required information.\n"
            "Now, based on this thought process, what would be the best way to "
            "search for and present information to answer this query comprehensively?"
        )
        return prompt

    def few_shot_examples(self) -> List[Dict[str, str]]:
        """Provide few-shot examples for the agent."""
        return [
            {"role": "user", "content": "What are the health benefits of green tea?"},
            {"role": "assistant", "content": "To answer this query, I would:\n1. Search for 'green tea health benefits scientific studies'\n2. Look for reputable health and nutrition websites\n3. Summarize the main benefits found in multiple sources\n4. Provide a concise list of benefits with brief explanations"},
            {"role": "user", "content": "Explain the process of photosynthesis in simple terms."},
            {"role": "assistant", "content": "For this query, I would:\n1. Search for 'photosynthesis explanation for beginners'\n2. Find educational resources and simple diagrams\n3. Break down the process into easy-to-understand steps\n4. Use analogies to make the concept more relatable"}
        ]

    def search_and_synthesize(self, query: str) -> str:
        """Perform web search and synthesize results using GPT."""
        try:
            # Generate chain of thought prompt
            cot_prompt = self.chain_of_thought_prompt(query)
            
            # Prepare messages for GPT
            messages = self.few_shot_examples()
            messages.append({"role": "user", "content": cot_prompt})
            
            # Get search strategy from GPT
            search_strategy = self._gpt_query(messages)
            
            # Perform web search
            search_results = self._web_search(query)
            
            # Prepare synthesis prompt
            synthesis_prompt = (
                f"Based on the search strategy: '{search_strategy}', "
                f"and the following search results: {search_results}, "
                f"please provide a comprehensive answer to the original query: '{query}'. "
                "Synthesize the information, cite sources where appropriate, "
                "and ensure the response is well-structured and easy to understand."
            )
            
            # Get final response from GPT
            final_response = self._gpt_query([{"role": "user", "content": synthesis_prompt}])
            
            return final_response
        except Exception as e:
            logger.error(f"Error in search_and_synthesize: {e}")
            return f"An error occurred while processing your query: {str(e)}"

# # Example usage
# if __name__ == "__main__":
#     agent = WebSearchAgent()
#     query = "What are the latest advancements in renewable energy technology?"
#     result = agent.search_and_synthesize(query)
#     print(result)