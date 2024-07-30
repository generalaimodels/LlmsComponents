import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Any
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class PPOAgent:
    def __init__(
        self,
        state_dim: int, 
        action_dim: int, 
        lr: float = 3e-4, 
        gamma: float = 0.99, 
        clip_epsilon: float = 0.2,
        epochs: int = 10, 
        batch_size: int = 64
    ):
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.policy_old.load_state_dict(self.policy.state_dict())  # Synchronize weights
        self.buffer = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy_old(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()

    def store_transition(self, transition: Tuple[np.ndarray, int, np.ndarray, float, float]):
        self.buffer.append(transition)

    def compute_returns(self, rewards: List[float], dones: List[bool], next_value: float) -> List[float]:
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - int(done))
            returns.insert(0, R)
        return returns
    
    def optimize(self) -> None:
        states, actions, rewards, dones, log_probs = zip(*self.buffer)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        
        next_state = states[-1]
        with torch.no_grad():
            next_value = self.value(next_state)

        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - self.value(states).detach()

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                sampled_indices = slice(i, i+self.batch_size)
                state_batch = states[sampled_indices]
                action_batch = actions[sampled_indices]
                advantage_batch = advantages[sampled_indices]
                old_log_prob_batch = old_log_probs[sampled_indices]
                
                new_log_probs = Categorical(self.policy(state_batch)).log_prob(action_batch)
                ratio = torch.exp(new_log_probs - old_log_prob_batch)

                surrogate1 = ratio * advantage_batch
                surrogate2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                value_loss = nn.MSELoss()(self.value(state_batch), returns[sampled_indices])
                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []

def train(env_name: str, num_episodes: int):
    env = gym.make(env_name)
    agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    for episode in range(num_episodes):
        try:
            state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
            print(f"Initial state: {state}")  # Add this line for debugging
            state = torch.tensor(state, dtype=torch.float32)
        except Exception as e:
            print(f"Error resetting environment: {e}")
            continue

        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            try:
                next_state, reward, done, _ = env.step(action) if isinstance(env.step(action), tuple) and len(env.step(action)) == 4 else (*env.step(action)[:4], {})
                next_state = torch.tensor(next_state, dtype=torch.float32)
            except Exception as e:
                print(f"Error during environment step: {e}")
                break

            agent.store_transition((state, action, reward, done, log_prob))
            state = next_state
            episode_reward += reward

        agent.optimize()
        print(f'Episode: {episode}, Reward: {episode_reward}')

    env.close()

if __name__ == "__main__":
    train(env_name='CartPole-v1', num_episodes=10)