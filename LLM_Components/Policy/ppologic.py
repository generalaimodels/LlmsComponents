import os
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class PPOMemory:
    def __init__(self):
        self.states: List[str] = []
        self.actions: List[int] = []
        self.probs: List[float] = []
        self.vals: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def generate_batches(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches
        )

    def store_memory(self, state: str, action: int, probs: float, vals: float, reward: float, done: bool) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()

class ActorCritic(nn.Module):
    def __init__(self, model_name: str, n_actions: int):
        super(ActorCritic, self).__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
        self.critic = nn.Linear(self.transformer.config.n_embd, 1)
        self.actor = nn.Linear(self.transformer.config.n_embd, n_actions)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.transformer(state)
        hidden_state = output.last_hidden_state[:, -1, :]
        value = self.critic(hidden_state)
        probs = torch.softmax(self.actor(hidden_state), dim=-1)
        return probs, value

class PPOAgent:
    def __init__(
        self,
        model_name: str,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.0003,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 64,
        n_epochs: int = 10
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor_critic = ActorCritic(model_name, n_actions)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=alpha)
        self.memory = PPOMemory()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_critic.to(self.device)

    def remember(self, state: str, action: int, probs: float, vals: float, reward: float, done: bool) -> None:
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation: str) -> Tuple[int, float, float]:
        state = self.tokenizer.encode(observation, return_tensors='pt').to(self.device)
        probs, value = self.actor_critic(state)

        action_probs = Categorical(probs)
        action = action_probs.sample()

        return action.item(), action_probs.log_prob(action).item(), value.item()

    def learn(self) -> None:
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches(self.batch_size)

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = self.tokenizer.batch_encode_plus([state_arr[i] for i in batch], padding=True, return_tensors='pt').to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                probs, critic_value = self.actor_critic(states.input_ids)

                critic_value = critic_value.squeeze()

                new_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.memory.clear_memory()

def train(env, agent: PPOAgent, n_episodes: int) -> None:
    for episode in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if len(agent.memory.states) == agent.batch_size:
                agent.learn()
            observation = new_observation
        print(f'Episode: {episode}, Score: {score}')

