import random
from typing import Dict, Any, Tuple,List
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
class AdvancedLLMEnvironment:
    def __init__(
        self,
        model1_name: str,
        model2_name: str,
        max_iterations: int = 1000,
        learning_rate: float = 1e-5,
    ) -> None:
        self.model1, self.tokenizer1 = self._load_model_and_tokenizer(model1_name)
        self.model2, self.tokenizer2 = self._load_model_and_tokenizer(model2_name)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.current_iteration = 0
        self.best_performance = float('-inf')
        self.prompting_template = self._initialize_prompting_template()

    @staticmethod
    def _load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @staticmethod
    def _initialize_prompting_template() -> str:
        return (
            "System: You are a helpful AI assistant.\n"
            "Human: Given the context: {context}\n"
            "Perform the following task: {task}\n"
            "Assistant: "
        )

    def reset(self) -> Dict[str, Any]:
        self.current_iteration = 0
        self.best_performance = float('-inf')
        return {"status": "Environment reset", "iteration": self.current_iteration}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_iteration += 1
        
        processed_action = self._process_action(action)
        
        response1 = self._generate_response(self.model1, self.tokenizer1, processed_action)
        response2 = self._generate_response(self.model2, self.tokenizer2, processed_action)
        
        reward = self._evaluate_responses(response1, response2)
        
        self.best_performance = max(self.best_performance, reward)
        
        done = self.current_iteration >= self.max_iterations or reward >= 0.95
        
        next_state = self._prepare_next_state(processed_action, response1, response2, reward)
        
        return next_state, reward, done, {
            "iteration": self.current_iteration,
            "best_performance": self.best_performance
        }

    def _process_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        processed_action = action.copy()
        processed_action["context"] = self._enhance_context(action.get("context", ""))
        processed_action["task"] = self._refine_task(action.get("task", ""))
        return processed_action

    @staticmethod
    def _enhance_context(context: str) -> str:
        # Implement context enhancement logic
        return f"Enhanced: {context}"

    @staticmethod
    def _refine_task(task: str) -> str:
        # Implement task refinement logic
        return f"Refined: {task}"

    def _generate_response(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        action: Dict[str, Any]
    ) -> str:
        chat_template = self._create_chat_template(action)
        inputs = tokenizer.apply_chat_template(chat_template, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=100, num_return_sequences=1)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _create_chat_template(self, action: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Context: {action['context']}"},
            {"role": "user", "content": f"Task: {action['task']}"},
        ]

    def _evaluate_responses(self, response1: str, response2: str) -> float:
        # Implement sophisticated response evaluation logic
        # This is a placeholder implementation
        return random.uniform(0, 1)

    @staticmethod
    def _prepare_next_state(
        action: Dict[str, Any],
        response1: str,
        response2: str,
        reward: float
    ) -> Dict[str, Any]:
        return {
            "previous_action": action,
            "model1_response": response1,
            "model2_response": response2,
            "previous_reward": reward,
        }

    def human_interface(self) -> None:
        print("Welcome to the Advanced LLM Environment Interface")
        while True:
            user_input = input("Enter your command (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            
            try:
                action = eval(user_input)
                if not isinstance(action, dict):
                    raise ValueError("Action must be a dictionary")
                
                next_state, reward, done, info = self.step(action)
                
                print(f"Next State: {next_state}")
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Info: {info}")
                
                if done:
                    print("Episode finished. Resetting environment.")
                    self.reset()
            
            except Exception as e:
                print(f"Error: {str(e)}")

    def advanced_analysis(self, num_episodes: int = 10) -> Dict[str, Any]:
        results = []
        for _ in tqdm(range(num_episodes), desc="Running episodes"):
            self.reset()
            episode_rewards = []
            while True:
                action = self._generate_random_action()
                _, reward, done, _ = self.step(action)
                episode_rewards.append(reward)
                if done:
                    break
            results.append(episode_rewards)
        
        return self._analyze_results(results)

    @staticmethod
    def _generate_random_action() -> Dict[str, str]:
        return {
            "context": f"Random context {random.randint(1, 100)}",
            "task": f"Random task {random.randint(1, 100)}",
        }

    @staticmethod
    def _analyze_results(results: List[List[float]]) -> Dict[str, Any]:
        episode_lengths = [len(episode) for episode in results]
        total_rewards = [sum(episode) for episode in results]
        
        return {
            "average_episode_length": sum(episode_lengths) / len(episode_lengths),
            "average_total_reward": sum(total_rewards) / len(total_rewards),
            "max_reward": max(max(episode) for episode in results),
            "min_reward": min(min(episode) for episode in results),
        }



class PPOPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class PPOValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)

class PPO:
    def __init__(
        self,
        env: AdvancedLLMEnvironment,
        state_dim: int,
        action_dim: int,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
    ):
        self.env = env
        self.policy = PPOPolicy(state_dim, action_dim)
        self.value = PPOValueFunction(state_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr_value)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def get_action(self, state: Dict[str, Any]) -> Tuple[Dict[str, str], torch.Tensor]:
        state_tensor = self._state_to_tensor(state)
        action_probs = torch.softmax(self.policy(state_tensor), dim=-1)
        dist = Categorical(action_probs)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        
        action = self._index_to_action(action_index.item())
        return action, log_prob

    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        # Convert state dictionary to a tensor representation
        # This is a simple example; you may need to adapt this based on your state structure
        return torch.tensor([
            len(state.get('previous_action', {}).get('context', '')),
            len(state.get('previous_action', {}).get('task', '')),
            len(state.get('model1_response', '')),
            len(state.get('model2_response', '')),
            state.get('previous_reward', 0),
        ], dtype=torch.float32)

    def _index_to_action(self, index: int) -> Dict[str, str]:
        # Convert action index to action dictionary
        # This is a simple example; you may need to adapt this based on your action space
        contexts = ["Context A", "Context B", "Context C"]
        tasks = ["Task X", "Task Y", "Task Z"]
        return {
            "context": contexts[index % len(contexts)],
            "task": tasks[index % len(tasks)],
        }

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + self.gamma * running_return
            returns.insert(0, running_return)
        return torch.tensor(returns)

    def update(self, trajectories: List[Dict[str, Any]]):
        states = torch.stack([self._state_to_tensor(t['state']) for t in trajectories])
        actions = torch.tensor([self._action_to_index(t['action']) for t in trajectories])
        old_log_probs = torch.stack([t['log_prob'] for t in trajectories])
        rewards = [t['reward'] for t in trajectories]
        returns = self.compute_returns(rewards)
        
        advantages = returns - self.value(states).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            for idx in range(0, states.size(0), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_old_log_probs = old_log_probs[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]

                action_probs = torch.softmax(self.policy(batch_states), dim=-1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(self.value(batch_states), batch_returns)

                entropy = dist.entropy().mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                loss.backward()
                self.optimizer_policy.step()
                self.optimizer_value.step()

    def _action_to_index(self, action: Dict[str, str]) -> int:
        # Convert action dictionary to index
        # This is a simple example; you may need to adapt this based on your action space
        contexts = ["Context A", "Context B", "Context C"]
        tasks = ["Task X", "Task Y", "Task Z"]
        return contexts.index(action['context']) * len(tasks) + tasks.index(action['task'])


def train_ppo(env: AdvancedLLMEnvironment, ppo: PPO, num_episodes: int = 1000):
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        trajectories = []
        episode_reward = 0
        
        while True:
            action, log_prob = ppo.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            trajectories.append({
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
            })
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        ppo.update(trajectories)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")
    
    return ppo

if __name__ == "__main__":
    # Initialize the environment
    env = AdvancedLLMEnvironment("gpt2", "gpt2-medium")
    
    # Define state and action dimensions
    state_dim = 5  # Adjust this based on your state representation
    action_dim = 9  # Adjust this based on your action space
    
    # Initialize PPO
    ppo = PPO(env, state_dim, action_dim)
    
    # Train PPO
    trained_ppo = train_ppo(env, ppo, num_episodes=1000)
    
    # Evaluate the trained model
    print("Evaluating trained model...")
    eval_episodes = 10
    total_reward = 0
    
    for _ in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = trained_ppo.get_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    average_reward = total_reward / eval_episodes
    print(f"Average reward over {eval_episodes} episodes: {average_reward}")
    
    # Run advanced analysis
    print("Running advanced analysis...")
    analysis_results = env.advanced_analysis()
    print("Advanced Analysis Results:")
    print(analysis_results)