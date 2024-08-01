import re
import warnings
from typing import List, Dict, Optional, Callable, Tuple, Union, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import PreTrainedModelWrapper
from datasets import Dataset
from torch.nn.parallel import DataParallel


from rich import print as rprint
from rich.text import Text

class TextHistory:
    """The TextHistory class keeps track of the history of an interaction between the language model and the environment."""

    def __init__(self, text: str, tokens: torch.LongTensor, system: bool = True):
        """
        Initialize TextHistory.

        Args:
            text (str): The text of the first segment.
            tokens (torch.LongTensor): The tokens of the first segment.
            system (bool, optional): Whether the first segment is a system or user segment.
        """
        self.system_spans = []
        self.text_spans = []
        self.token_spans = []
        self.token_masks = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.text = ""
        self.tokens = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.completed = False
        self.truncated = False
        self.reward = 0.0

        self.prompt_color = "black on grey85"
        self.system_color = "black on cyan3"
        self.model_color = "black on deep_sky_blue1"
        self.reward_color = "black on plum1"

        self.append_segment(text, tokens, system=system)

    def append_segment(self, text: str, tokens: torch.LongTensor, system: bool = True):
        """
        Append a new segment to the history.

        Args:
            text (str): The text of the new segment.
            tokens (torch.LongTensor): The tokens of the new segment.
            system (bool, optional): Whether the new segment is a system or user segment.
        """

        if not text or not len(tokens):
            raise ValueError("Can't append empty text or token list to history.")

        original_text_length = len(self.text)

        self.text += text
        self.text_spans.append((original_text_length, len(self.text)))
        self.system_spans.append(system)

        original_token_length = len(self.tokens)

        self.tokens = torch.cat((self.tokens, tokens))
        self.token_masks = torch.cat((self.token_masks,
                                      torch.zeros_like(tokens) if system else torch.ones_like(tokens)))
        self.token_spans.append((original_token_length, len(self.tokens)))

    def complete(self, truncated: bool = False):
        """
        Mark the history as completed.
        """
        self.completed = True
        self.truncated = truncated

    @property
    def last_text_segment(self) -> str:
        """
        Get the last text segment.
        """
        start, end = self.text_spans[-1]
        return self.text[start:end]

    def split_query_response_tokens(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the tokens into query and response tokens.
        """
        split_index = self.token_spans[0][1]
        query = self.tokens[:split_index]
        response = self.tokens[split_index:]
        mask = self.token_masks[split_index:]

        return query, response, mask

    def show_text(self, show_legend: bool = False):
        """
        Print the text history with rich.

        Args:
            show_legend (bool, optional): Whether to show the color legend or not.
        """
        if not is_rich_available():
            warnings.warn("Install rich to display text")
            return

        text = Text(self.text)
        text.stylize(self.prompt_color, self.text_spans[0][0], self.text_spans[1][0])

        for i, (start, end) in enumerate(self.text_spans[1:]):
            color = self.system_color if self.system_spans[i + 1] else self.model_color
            text.stylize(color, start, end)

        text.append(f"\n\nReward: {self.reward}", style=self.reward_color)
        rprint(text)

        if show_legend:
            self.show_colour_legend()

    def show_tokens(self, tokenizer: PreTrainedTokenizer, show_legend: bool = False):
        """
        Print the history tokens with rich.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer used for decoding.
            show_legend (bool, optional): Whether to show the color legend or not.
        """
        if not is_rich_available():
            warnings.warn("Install rich to display tokens")
            return

        text = Text()
        prompt_end = self.token_spans[0][1]

        for i, (token, mask) in enumerate(zip(self.tokens, self.token_masks)):
            token_text = tokenizer.convert_ids_to_tokens(token.item())
            color = (self.prompt_color if i < prompt_end else
                     self.system_color if mask == 0 else
                     self.model_color)
            text.append(token_text, style=color).append(" ")

        text.append(f"\n\nReward: {self.reward}", style=self.reward_color)
        rprint(text)

        if show_legend:
            self.show_colour_legend()

    def show_colour_legend(self):
        """
        Print the color legend with rich.
        """
        if not is_rich_available():
            warnings.warn("Install rich to display color legend")
            return

        text = Text("\n\n(Colour Legend: ")
        text.append("Prompt", style=self.prompt_color).append("|")
        text.append("System", style=self.system_color).append("|")
        text.append("Model", style=self.model_color).append("|")
        text.append("Reward", style=self.reward_color).append(")")
        rprint(text)


class TextEnvironment:
    """
    The TextEnvironment enables interaction of a LLM with an environment using tools.
    """

    def __init__(
        self,
        model: PreTrainedModelWrapper,
        tokenizer: PreTrainedTokenizer,
        tools: Union[Dict[str, Callable], List[Callable]],
        reward_fn: Callable[[List[str]], List[float]],
        prompt: str,
        max_turns: int = 4,
        max_tool_response: int = 100,
        max_length: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TextEnvironment.

        Args:
            model (PreTrainedModelWrapper): The model to use for generation.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for generation.
            tools (Union[Dict[str, Callable], List[Callable]]): A dictionary or list of tools to use for interaction.
            reward_fn (Callable[[str], float]): A function that takes a list of strings and returns a list of rewards.
            prompt (str): The base prompt to use for generation.
            max_turns (int, optional): The maximum number of turns to allow.
            max_tool_response (int, optional): The maximum number of characters to allow in a tool response.
            max_length (int, optional): The maximum number of tokens to allow in an episode.
            generation_kwargs (dict, optional): Additional keyword arguments for the model's generate method.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.tools = {tool.__class__.__name__: tool for tool in tools} if isinstance(tools, list) else tools
        self.reward_fn = reward_fn
        self.max_length = max_length
        self.max_turns = max_turns
        self.max_tool_response = max_tool_response
        self.generation_kwargs = generation_kwargs or {}

        self.request_token = "<request>"
        self.call_token = "<call>"
        self.response_token = "<response>"
        self.submit_token = "<submit>"

        self.is_encoder_decoder = hasattr(model, "is_encoder_decoder")
        self.current_device = self._extract_device()

    def run(self, queries: List[str], **reward_kwargs) -> Tuple[List[torch.Tensor], List[torch.Tensor], 
                                                                 List[torch.Tensor], List[float], List[TextHistory]]:
        """
        Run the environment on a list of queries.

        Args:
            queries (list[str]): A list of queries to run the model in the environment on.

        Returns:
            queries, responses, masks (all torch.Tensor): Token tensors for queries and responses along with masks.
            rewards (list[float]): List of rewards for each interaction.
            histories (list[TextHistory]): List of history objects containing interaction data.
        """
        turns = 0

        queries = [self.prompt + task for task in queries]
        queries_tokens = [self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.current_device) for query in queries]

        histories = [TextHistory(q, qt, system=True) for q, qt in zip(queries, queries_tokens)]

        while any(not history.completed for history in histories) and turns < self.max_turns:
            histories = self.generate(histories)
            histories = self.tasks_end_check(histories)
            for i in range(len(histories)):
                histories[i] = self.step(histories[i])
            histories = self.tasks_end_check(histories, model_turn=False)
            turns += 1
        self.compute_reward(histories, **reward_kwargs)

        queries, responses, masks = map(list, zip(*[history.split_query_response_tokens() for history in histories]))
        rewards = [history.reward for history in histories]
        return queries, responses, masks, rewards, histories

    def step(self, history: TextHistory) -> TextHistory:
        """
        Step the environment forward one turn.

        Args:
            history (TextHistory): The history to step forward.

        Returns:
            TextHistory: The updated history after taking a step.
        """
        truncated, ended = self.task_end_check(history)
        if ended:
            history.complete(truncated=truncated)

        if history.completed:
            return history

        tool, query = self.parse_tool_call(history.last_text_segment)
        if tool is None or query is None:
            response = f"Unknown tool call: {history.last_text_segment}"
        else:
            if tool not in self.tools:
                response = f"Unknown tool {tool}."
            else:
                try:
                    response = self.tools[tool](query)
                except Exception as error:
                    response = f"Tool error: {str(error)}"

        if len(response) > self.max_tool_response:
            response = response[: (self.max_tool_response - 3)] + "..."

        history.append_segment(
            response + self.response_token,
            self.tokenizer(response + self.response_token, return_tensors="pt").input_ids[0].to(self.current_device),
            system=True,
        )

        return history

    def parse_tool_call(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse request string. Expected format: <request><tool_name>query<call>

        Args:
            text (str): The text to parse.

        Returns:
            Tuple[Optional[str], Optional[str]]: The tool name and query, if found.
        """
        result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text, re.DOTALL)

        if result is None:
            return None, None

        extracted_text = result.group()
        result = re.search(r"<(.*?)>", extracted_text)

        if result is None:
            return None, None

        tool = result.group(1)
        query = ">".join(extracted_text.split(">")[1:])

        return tool, query

    def compute_reward(self, histories: List[TextHistory], **reward_kwargs) -> List[TextHistory]:
        """
        Compute the reward for a list of histories.

        Args:
            histories (List[TextHistory]): List of text interaction histories.

        Returns:
            List[TextHistory]: Updated list of histories with computed rewards.
        """
        rewards = self.reward_fn([history.last_text_segment for history in histories], **reward_kwargs)
        for history, reward in zip(histories, rewards):
            history.reward = reward
        return histories

    def generate(self, histories: List[TextHistory]) -> List[TextHistory]:
        """
        Generate responses for a list of histories.

        Args:
            histories (List[TextHistory]): List of text interaction histories.

        Returns:
            List[TextHistory]: Updated list of histories with generated responses.
        """
        active_indices = [i for i, history in enumerate(histories) if not history.completed]
        query_tensors = [histories[i].tokens for i in active_indices]
        response_tensors = self._generate_batched(query_tensors)

        for i, response_tensor in zip(active_indices, response_tensors):
            response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            histories[i].append_segment(
                response_text + self.submit_token,
                response_tensor,
                system=False
            )

        return histories

    def _generate_batched(self, query_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Generate batched responses.

        Args:
            query_tensors (List[torch.Tensor]): List of query tensors.

        Returns:
            List[torch.Tensor]: List of generated response tensors.
        """
        inputs = self.tokenizer.pad({"input_ids": query_tensors}, return_tensors="pt").to(self.current_device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_kwargs)
        return [output for output in outputs]

    def _extract_device(self) -> torch.device:
        """
        Extract the current device from the model.

        Returns:
            torch.device: The device of the model.
        """
        if isinstance(self.model, DataParallel):
            return self.model.module.device
        return self.model.device

    def task_end_check(self, history: TextHistory) -> Tuple[bool, bool]:
        """
        Check if the task has ended.

        Args:
            history (TextHistory): The history to check.

        Returns:
            Tuple[bool, bool]: Whether the task is truncated and/or ended.
        """
        if self.max_length and len(history.tokens) > self.max_length:
            return True, True
        if self.submit_token in history.text:
            return False, True
        return False, False

    def tasks_end_check(self, histories: List[TextHistory], model_turn: bool = True) -> List[TextHistory]:
        """
        Check if all tasks in a list of histories have ended.

        Args:
            histories (List[TextHistory]): List of text interaction histories.
            model_turn (bool, optional): Whether it is the model's turn or the user's turn.

        Returns:
            List[TextHistory]: Updated list of histories.
        """
        for history in histories:
            if not history.completed:
                truncated, ended = self.task_end_check(history)
                if ended:
                    history.complete(truncated=truncated)
        return histories

# Utility function to check if rich is available
def is_rich_available() -> bool:
    try:
        import rich
        return True
    except ImportError:
        return False