import csv
import importlib
import os
import random
import sys
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Sources.Core.device import resolve_torch_device
from Sources.RL.Buffers import NStepReplayBuffer
from Sources.RL.Networks import MultiHeadQNetwork, QNetwork


class RolloutBuffer:
    def __init__(self):
        # No deques, just simple flat lists!
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.is_terminals = []

    def push(self, state, action, log_prob, value, reward, done):
        # We just append the raw data sequentially
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        
    def clear(self):
        # Clear everything for the next rollout
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.values[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(ActorCritic, self).__init__()

        self.shared = self._build_network(state_dim, hidden_dims)

        last_hidden_dim = hidden_dims[-1] if hidden_dims else state_dim

        self.actor = nn.Sequential(
            nn.Linear(last_hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(last_hidden_dim, 1)

    def _build_network(self, input_dim, hidden_dims):
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        shared = self.shared(x)
        action_probs = self.actor(shared)
        state_value = self.critic(shared)

        return action_probs, state_value
    
    def evaluate(self, states, actions):
        raise NotImplementedError

class BaseWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger
        
        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_meta
        self.action_dim = cfg.action_dim_meta
        self.hidden_dims = cfg.hidden_dims

        self.K_epochs = cfg.K_epochs
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.entropy_coef = cfg.coef_entropy
        self.clip_ratio = cfg.clip_ratio
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        
        self.device = resolve_torch_device()

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_dims)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=cfg.learning_rate
        )

        self.loss_fn = nn.MSELoss()
        self.nb_interval = cfg.nb_interval

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, state_value = self.policy(state)

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        return action.item(), action_log_prob.item(), state_value.item()

    def remember(self, state, action, log_prob, value, reward, done):
        self.buffer.push(state, action, log_prob, value, reward, done)
    
    def train_step(self):
        self.step += 1
        if self.step % self.nb_interval == 0 and \
           len(self.buffer) >= self.batch_size:
            self.update(self.buffer)

    def compute_advantages(
        self, 
        rewards, 
        values, 
        is_terminals, 
        last_value, 
        gamma=0.99, 
        gae_lambda=0.95
    ):
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[i + 1]

            # 1. Calculate the 1-step Temporal Difference (TD) Error (called Delta)
            delta = rewards[i] + gamma * next_value * (1 - is_terminals[i]) - values[i]

            # 2. Accumulate the GAE advantage using the recursive formula
            gae = delta + gamma * gae_lambda * (1 - is_terminals[i]) * gae

            # 3. Store the computed advantage for the current timestep
            advantages.insert(0, gae)

        # The actual Return is just the Advantage + the Critic's predicted Value
        returns = [adv + val for adv, val in zip(advantages, values)]

        return advantages, returns

    def update(self, rollout_buffer):
        
        # 1. Calculate advantages and returns (GAE) using the buffer data
        last_value = 0 if rollout_buffer.is_terminals[-1] else self.policy.critic(
            torch.tensor(rollout_buffer.states[-1], dtype=torch.float32).unsqueeze(0).to(self.device)
        ).item()

        advantages, returns = self.compute_advantages(
            rewards=rollout_buffer.rewards,
            values=rollout_buffer.values,
            is_terminals=rollout_buffer.is_terminals,
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # 2. Convert lists to tensors for training
        states = torch.tensor(rollout_buffer.states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(rollout_buffer.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(rollout_buffer.log_probs, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
    
        for _ in range(self.K_epochs):

            # 3. Evaluate the current policy on the sampled states and actions
            action_probs, state_values = self.policy(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)

            # 4. Calculate the ratio of new and old action probabilities
            ratios = torch.exp(new_log_probs - old_log_probs)

            # 5. Compute the surrogate loss using the PPO clipping mechanism
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 6. Compute the critic loss (value function loss)
            critic_loss = self.loss_fn(state_values.squeeze(), returns)

            # 7. Compute the total loss with an entropy bonus for exploration
            entropy_loss = action_dist.entropy().mean()
            total_loss = actor_loss + self.cfg.coef_critic * critic_loss - self.entropy_coef * entropy_loss

            # 8. Perform backpropagation and optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm=0.5)
            self.optimizer.step()

        # 4. Crucial Step: Clear the memory for the next rollout!
        rollout_buffer.clear()


class EnhWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_ctrl
        self.action_dim = cfg.action_dim_ctrl

        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        self.device = resolve_torch_device()

        self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)

        self.policy_net = MultiHeadQNetwork(
            self.state_dim,
            self.action_dim,
            num_heads=4
        ).to(self.device)
        self.target_net = MultiHeadQNetwork(
            self.state_dim,
            self.action_dim,
            num_heads=4
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=cfg.learning_rate_decay
        )

        self.loss_fn = nn.MSELoss()
        self.nb_interval = cfg.nb_interval

    def select_action(self, state):
        if random.random() < self.epsilon:
            return [random.randint(0, self.action_dim - 1) for _ in range(4)]

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.policy_net(state)   # (1, 4, actions)
            actions = qvals.argmax(dim=2).squeeze(0)
            return actions.tolist()

    def remember(self, s, action, r, ns, done):
        self.buffer.push(s, action, r, ns, done)

    def train_step(self):
        self.step += 1
        if self.step % self.nb_interval == 0 and \
           len(self.buffer) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s = torch.tensor(np.stack(s), dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.stack(ns), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.stack(a), dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32).to(self.device).unsqueeze(1)

        # ---------- target ----------
        with torch.no_grad():
            next_q = self.target_net(ns)
            q_max_next = next_q.max(dim=2)[0]
            n_step_gamma = self.gamma ** self.n_step
            q_target = r + n_step_gamma * q_max_next * (1 - d)

        # ---------- expected ----------
        q_expected = self.policy_net(s).gather(2, a.unsqueeze(2)).squeeze(2)

        loss = self.loss_fn(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target()

        self.debugger.log('train_loss_enh', loss.item())
        self.debugger.log('epsilon_enh', self.epsilon)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        for tparam, pparam in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tparam.data.copy_(
                tparam.data * (1.0 - self.tau) + pparam.data * self.tau
            )

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_step(self):
        self.step = 0
