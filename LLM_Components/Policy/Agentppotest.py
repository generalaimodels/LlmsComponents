import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from collections import namedtuple
import numpy as np

# Hyperparameters
EPSILON = 0.2
GAMMA = 0.99
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 64
ENTROPY_COEFFICIENT = 0.01
VALUE_LOSS_COEFFICIENT = 0.5
NUM_EPISODES = 1000
Trajectory = namedtuple('Trajectory', ['states', 'actions', 'rewards', 'dones', 'log_probs'])

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
    def __init__(self, state_dim: int, action_dim: int, lr: float = LEARNING_RATE, gamma: float = GAMMA, epsilon: float = EPSILON):
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer = []

        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            action_probs = self.policy_old(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()

    def store_transition(self, transition: tuple) -> None:
        self.buffer.append(transition)

    def compute_returns(self) -> torch.Tensor:
        returns, R = [], 0
        for _, _, reward, done, _ in reversed(self.buffer):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def optimize(self) -> None:
        if not self.buffer:
            return

        transitions = Trajectory(*zip(*self.buffer))
        states = torch.stack(transitions.states)
        actions = torch.tensor(transitions.actions)
        old_log_probs = torch.tensor(transitions.log_probs)
        returns = self.compute_returns().float()

        advantages = returns - self.value(states).view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # Normalize advantages

        for _ in range(NUM_EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                sampled_indices = slice(i, i + BATCH_SIZE)
                state_batch = states[sampled_indices]
                action_batch = actions[sampled_indices]
                advantage_batch = advantages[sampled_indices]
                old_log_prob_batch = old_log_probs[sampled_indices]

                new_log_probs = Categorical(self.policy(state_batch)).log_prob(action_batch)
                ratio = torch.exp(new_log_probs - old_log_prob_batch)

                surrogate1 = ratio * advantage_batch
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = nn.MSELoss()(self.value(state_batch).view(-1), returns[sampled_indices])

                entropy_bonus = Categorical(self.policy(state_batch)).entropy().mean()

                total_loss = policy_loss + VALUE_LOSS_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy_bonus

                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                total_loss.backward()
                self.optimizer_policy.step()
                self.optimizer_value.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []

def train(env_name: str, num_episodes: int):
    env = gym.make(env_name)
    agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle tuple state
        print(f"Initial state: {state}")
        state = torch.tensor(state, dtype=torch.float32)

        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            step_output = env.step(action)
            if len(step_output) == 4:
                next_state, reward, done, _ = step_output
            else:
                next_state, reward, done, _, _ = step_output
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Handle tuple state
            next_state = torch.tensor(next_state, dtype=torch.float32)

            agent.store_transition((state, action, reward, done, log_prob))
            state = next_state
            episode_reward += reward

        agent.optimize()
        print(f'Episode: {episode}, Reward: {episode_reward}')

    env.close()

if __name__ == "__main__":
    train(env_name='CartPole-v1', num_episodes=NUM_EPISODES)