import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np

# Hyperparameters
EPSILON = 0.2
GAMMA = 0.99
LEARNING_RATE = 0.002
NUM_EPOCHS = 4
BATCH_SIZE = 32
ENTROPY_COEFFICIENT = 0.01
VALUE_LOSS_COEFFICIENT = 0.5
NUM_EPISODES = 1000

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=LEARNING_RATE)
        self.buffer = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def update(self):
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.buffer)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.stack(old_log_probs)

        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(self.value(states).detach())):
            R = reward + GAMMA * R * (1 - done)
            returns.insert(0, R)
            advantage = R - value.item()
            advantages.insert(0, advantage)
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(NUM_EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                batch_states = states[i:i+BATCH_SIZE]
                batch_actions = actions[i:i+BATCH_SIZE]
                batch_log_probs = old_log_probs[i:i+BATCH_SIZE]
                batch_returns = returns[i:i+BATCH_SIZE]
                batch_advantages = advantages[i:i+BATCH_SIZE]

                # Compute new action probabilities
                new_action_probs = self.policy(batch_states)
                dist = Categorical(new_action_probs)
                new_log_probs = dist.log_prob(batch_actions)

                # Compute ratio and surrogate loss
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_pred = self.value(batch_states).squeeze()
                value_loss = nn.MSELoss()(value_pred, batch_returns)

                # Compute entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + VALUE_LOSS_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy

                # Update policy and value networks
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                loss.backward()
                self.optimizer_policy.step()
                self.optimizer_value.step()

        self.buffer = []

def train(env_name, num_episodes):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            state = next_state
            episode_reward += reward

            if done:
                agent.update()
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    train('CartPole-v1', NUM_EPISODES)