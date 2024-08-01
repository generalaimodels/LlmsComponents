import re
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList

@dataclass
class TextSegment:
    text: str
    tokens: Tensor
    is_system: bool

@dataclass
class TextHistory:
    segments: List[TextSegment] = field(default_factory=list)
    completed: bool = False
    truncated: bool = False
    reward: float = 0.0

    def append_segment(self, segment: TextSegment) -> None:
        self.segments.append(segment)

    @property
    def last_segment(self) -> TextSegment:
        return self.segments[-1]

    @property
    def full_text(self) -> str:
        return "".join(segment.text for segment in self.segments)

    @property
    def full_tokens(self) -> Tensor:
        return torch.cat([segment.tokens for segment in self.segments])

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_strings: List[str], tokenizer: PreTrainedTokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> bool:
        decoded = self.tokenizer.batch_decode(input_ids)
        return any(any(stop_string in text for stop_string in self.stop_strings) for text in decoded)

class TextEnvironment:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tools: Dict[str, Callable],
        reward_fn: Callable[[List[str]], List[float]],
        prompt: str,
        max_turns: int = 4,
        max_tool_response: int = 100,
        max_length: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.reward_fn = reward_fn
        self.prompt = prompt
        self.max_turns = max_turns
        self.max_tool_response = max_tool_response
        self.max_length = max_length
        self.generation_kwargs = generation_kwargs or {}

        self.request_token = "<request>"
        self.call_token = "<call>"
        self.response_token = "<response>"
        self.submit_token = "<submit>"

    def run(self, queries: List[str]) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[float], List[TextHistory]]:
        histories = self._initialize_histories(queries)
        
        for _ in range(self.max_turns):
            if all(history.completed for history in histories):
                break
            
            self._generate_responses(histories)
            self._process_tool_calls(histories)
            self._check_completions(histories)

        self._compute_rewards(histories)
        return self._prepare_output(histories)

    def _initialize_histories(self, queries: List[str]) -> List[TextHistory]:
        histories = []
        for query in queries:
            full_prompt = f"{self.prompt}{query}"
            tokens = self.tokenizer(full_prompt, return_tensors="pt").input_ids[0].to(self.model.device)
            segment = TextSegment(full_prompt, tokens, is_system=True)
            histories.append(TextHistory([segment]))
        return histories

    def _generate_responses(self, histories: List[TextHistory]) -> None:
        active_histories = [h for h in histories if not h.completed]
        if not active_histories:
            return

        input_tensors = [h.full_tokens for h in active_histories]
        stopping_criteria = CustomStoppingCriteria([self.call_token, self.submit_token], self.tokenizer)
        
        outputs = self.model.generate(
            input_ids=torch.stack(input_tensors),
            stopping_criteria=StoppingCriteriaList([stopping_criteria]),
            **self.generation_kwargs
        )

        for history, output in zip(active_histories, outputs):
            new_tokens = output[len(history.full_tokens):]
            new_text = self.tokenizer.decode(new_tokens)
            history.append_segment(TextSegment(new_text, new_tokens, is_system=False))

    def _process_tool_calls(self, histories: List[TextHistory]) -> None:
        for history in histories:
            if history.completed:
                continue

            tool, query = self._parse_tool_call(history.last_segment.text)
            if tool is None or query is None:
                response = f"Error: Invalid tool call format"
            elif tool not in self.tools:
                response = f"Error: Unknown tool '{tool}'"
            else:
                try:
                    response = self.tools[tool](query)
                    response = response[:self.max_tool_response]
                except Exception as e:
                    response = f"Error: {str(e)}"

            response_text = f"{response}{self.response_token}"
            response_tokens = self.tokenizer(response_text, return_tensors="pt").input_ids[0].to(self.model.device)
            history.append_segment(TextSegment(response_text, response_tokens, is_system=True))

    def _parse_tool_call(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        pattern = f"{self.request_token}<(.+?)>(.+?){self.call_token}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None

    def _check_completions(self, histories: List[TextHistory]) -> None:
        for history in histories:
            if history.completed:
                continue

            if self.max_length and len(history.full_tokens) > self.max_length:
                history.completed = True
                history.truncated = True
            elif self.submit_token in history.last_segment.text:
                history.completed = True

    def _compute_rewards(self, histories: List[TextHistory]) -> None:
        rewards = self.reward_fn([h.full_text for h in histories])
        for history, reward in zip(histories, rewards):
            history.reward = reward

    def _prepare_output(self, histories: List[TextHistory]) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[float], List[TextHistory]]:
        queries = [h.segments[0].tokens for h in histories]
        responses = [torch.cat([s.tokens for s in h.segments[1:]]) for h in histories]
        masks = [torch.cat([torch.ones_like(s.tokens) if not s.is_system else torch.zeros_like(s.tokens) for s in h.segments[1:]]) for h in histories]
        rewards = [h.reward for h in histories]
        return queries, responses, masks, rewards, histories
    



import asyncio
from typing import List, Dict, Any, Callable, Coroutine, Optional
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from pydantic import BaseModel
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
import ray

@dataclass
class EnvironmentConfig:
    max_turns: int
    max_tool_response: int
    max_length: Optional[int]
    request_token: str
    call_token: str
    response_token: str
    submit_token: str

@dataclass
class TextSegment:
    text: str
    tokens: Tensor
    is_system: bool

class TextHistory(BaseModel):
    segments: List[TextSegment] = field(default_factory=list)
    completed: bool = False
    truncated: bool = False
    reward: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def append_segment(self, segment: TextSegment) -> None:
        self.segments.append(segment)

    @property
    def last_segment(self) -> TextSegment:
        return self.segments[-1]

    @property
    def full_text(self) -> str:
        return "".join(segment.text for segment in self.segments)

    @property
    def full_tokens(self) -> Tensor:
        return torch.cat([segment.tokens for segment in self.segments])

class ToolExecutor:
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        self.executor = ThreadPoolExecutor()

    async def execute(self, tool_name: str, query: str) -> str:
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(self.executor, self.tools[tool_name], query)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

@ray.remote
class ModelWorker:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: EnvironmentConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def generate(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def tokenize(self, text: str) -> Dict[str, Tensor]:
        return self.tokenizer(text, return_tensors="pt")

    def decode(self, token_ids: Tensor) -> str:
        return self.tokenizer.decode(token_ids)

class TextEnvironment:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tools: Dict[str, Callable],
        reward_fn: Callable[[List[str]], List[float]],
        prompt: str,
        config: EnvironmentConfig,
        num_workers: int = 4
    ):
        self.config = config
        self.prompt = prompt
        self.reward_fn = reward_fn
        self.tool_executor = ToolExecutor(tools)
        
        ray.init(ignore_reinit_error=True)
        self.workers = [ModelWorker.remote(model, tokenizer, config) for _ in range(num_workers)]

    async def run(self, queries: List[str]) -> List[TextHistory]:
        histories = await self._initialize_histories(queries)
        
        for _ in range(self.config.max_turns):
            if all(history.completed for history in histories):
                break
            
            await self._generate_responses(histories)
            await self._process_tool_calls(histories)
            self._check_completions(histories)

        await self._compute_rewards(histories)
        return histories

    async def _initialize_histories(self, queries: List[str]) -> List[TextHistory]:
        histories = []
        tokenize_tasks = [worker.tokenize.remote(f"{self.prompt}{query}") for worker, query in zip(self.workers, queries)]
        tokenized_results = await asyncio.gather(*[ray.get(task) for task in tokenize_tasks])

        for query, tokenized in zip(queries, tokenized_results):
            full_prompt = f"{self.prompt}{query}"
            segment = TextSegment(full_prompt, tokenized['input_ids'][0], is_system=True)
            histories.append(TextHistory(segments=[segment]))
        return histories

    async def _generate_responses(self, histories: List[TextHistory]) -> None:
        active_histories = [h for h in histories if not h.completed]
        if not active_histories:
            return

        input_tensors = [h.full_tokens for h in active_histories]
        attention_masks = [torch.ones_like(tensor) for tensor in input_tensors]

        generate_tasks = [
            worker.generate.remote(input_ids=input_tensor.unsqueeze(0), 
                                   attention_mask=attention_mask.unsqueeze(0))
            for worker, input_tensor, attention_mask in zip(self.workers, input_tensors, attention_masks)
        ]
        outputs = await asyncio.gather(*[ray.get(task) for task in generate_tasks])

        decode_tasks = [worker.decode.remote(output[0]) for worker, output in zip(self.workers, outputs)]
        decoded_outputs = await asyncio.gather(*[ray.get(task) for task in decode_tasks])

        for history, output, decoded in zip(active_histories, outputs, decoded_outputs):
            new_tokens = output[0][len(history.full_tokens):]
            new_text = decoded[len(history.full_text):]
            history.append_segment(TextSegment(new_text, new_tokens, is_system=False))

    async def _process_tool_calls(self, histories: List[TextHistory]) -> None:
        tool_calls = []
        for history in histories:
            if history.completed:
                continue
            tool, query = self._parse_tool_call(history.last_segment.text)
            tool_calls.append((tool, query))

        responses = await asyncio.gather(*[self.tool_executor.execute(tool, query) for tool, query in tool_calls if tool and query])

        for history, response in zip(histories, responses):
            if history.completed:
                continue
            response_text = f"{response[:self.config.max_tool_response]}{self.config.response_token}"
            tokenize_task = self.workers[0].tokenize.remote(response_text)
            tokenized = await ray.get(tokenize_task)
            history.append_segment(TextSegment(response_text, tokenized['input_ids'][0], is_system=True))

    def _parse_tool_call(self, text: str) -> tuple[Optional[str], Optional[str]]:
        import re
        pattern = f"{self.config.request_token}<(.+?)>(.+?){self.config.call_token}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None

    def _check_completions(self, histories: List[TextHistory]) -> None:
        for history in histories:
            if history.completed:
                continue
            if self.config.max_length and len(history.full_tokens) > self.config.max_length:
                history.completed = True
                history.truncated = True
            elif self.config.submit_token in history.last_segment.text:
                history.completed = True

    async def _compute_rewards(self, histories: List[TextHistory]) -> None:
        texts = [h.full_text for h in histories]
        rewards = await asyncio.to_thread(self.reward_fn, texts)
        for history, reward in zip(histories, rewards):
            history.reward = reward

class EnvironmentManager:
    def __init__(self, environment: TextEnvironment):
        self.environment = environment
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/run")
        async def run_environment(queries: List[str]):
            histories = await self.environment.run(queries)
            return [self._serialize_history(h) for h in histories]

    def _serialize_history(self, history: TextHistory):
        return {
            "full_text": history.full_text,
            "reward": history.reward,
            "completed": history.completed,
            "truncated": history.truncated,
            "segments": [
                {
                    "text": seg.text,
                    "is_system": seg.is_system
                } for seg in history.segments
            ]
        }

    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

@dataclass
class BatchProcessor:
    environment: TextEnvironment
    batch_size: int = 32

    async def process_batches(self, all_queries: List[str]) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(all_queries), self.batch_size):
            batch = all_queries[i:i+self.batch_size]
            batch_results = await self.environment.run(batch)
            results.extend(batch_results)
        return results

def create_environment(model_path: str, tokenizer_path: str, tools: Dict[str, Callable], 
                       reward_fn: Callable, prompt: str, config: EnvironmentConfig, 
                       num_workers: int = 4) -> TextEnvironment:
    model = ray.remote(num_gpus=1)(PreTrainedModel).remote().from_pretrained(model_path)
    tokenizer = PreTrainedTokenizer.from_pretrained(tokenizer_path)
    return TextEnvironment(model, tokenizer, tools, reward_fn, prompt, config, num_workers)

if __name__ == "__main__":
    # Example usage
    config = EnvironmentConfig(
        max_turns=4,
        max_tool_response=100,
        max_length=1024,
        request_token="<request>",
        call_token="<call>",
        response_token="<response>",
        submit_token="<submit>"
    )

    def dummy_tool(query: str) -> str:
        return f"Processed: {query}"

    def dummy_reward_fn(texts: List[str]) -> List[float]:
        return [len(text) for text in texts]

    tools = {"dummy_tool": dummy_tool}
    env = create_environment(
        model_path="gpt2",
        tokenizer_path="gpt2",
        tools=tools,
        reward_fn=dummy_reward_fn,
        prompt="This is a test prompt. ",
        config=config
    )

    manager = EnvironmentManager(env)
    manager.run_server()