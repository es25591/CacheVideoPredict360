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

    def __len__(self):
        return len(self.states)

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

class MultiHeadActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=4, hidden_dim=256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 4 actor heads
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim)
            for _ in range(num_heads)
        ])

        # single critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.shared(x)

        logits = [head(shared) for head in self.actor_heads]
        value = self.critic(shared)

        return logits, value
    
class BaseWorker:
    def __init__(self, cfg, debugger=None):
        self.device = resolve_torch_device()
        
        self.cfg = cfg
        self.debugger = debugger
        
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_meta
        self.action_dim = cfg.action_dim_meta
        self.hidden_dims = cfg.hidden_dims

        self.K_epochs = cfg.K_epochs
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.entropy_coef = cfg.coef_entropy
        self.value_loss_coef = cfg.value_loss_coef
        self.clip_ratio = cfg.clip_ratio
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
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
        if len(self.buffer) >= self.n_step:
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
        if rollout_buffer.is_terminals[-1]:
            last_value = 0.0
        else:
            last_state = torch.tensor(
                rollout_buffer.states[-1], dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, last_state_value = self.policy(last_state)
            last_value = last_state_value.item()

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
        old_values = torch.tensor(rollout_buffer.values, dtype=torch.float32).to(self.device)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        
        for _ in range(self.K_epochs):
            
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_old_values = old_values[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Forward pass
                action_probs, state_values = self.policy(batch_states)
                state_values = state_values.view(-1)

                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # --------------------------------------------------
                # 6️⃣  PPO Clipped Objective (Actor)
                # --------------------------------------------------
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1 - self.clip_ratio,
                    1 + self.clip_ratio
                ) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                # --------------------------------------------------
                # 7️⃣  Value Function Clipping (IMPORTANT FIX)
                # --------------------------------------------------
                value_pred_clipped = batch_old_values + torch.clamp(
                    state_values - batch_old_values,
                    -self.clip_ratio,
                    self.clip_ratio
                )

                value_losses = (state_values - batch_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)

                critic_loss = 0.5 * torch.max(
                    value_losses,
                    value_losses_clipped
                ).mean()

                # --------------------------------------------------
                # 8️⃣  Total Loss
                # --------------------------------------------------
                total_loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        # --------------------------------------------------
        # 9️⃣  Clear buffer after update
        # --------------------------------------------------
        rollout_buffer.clear()

class EnhWorker:
    def __init__(self, cfg, debugger=None):

        self.cfg = cfg
        self.debugger = debugger
        self.device = resolve_torch_device()

        self.state_dim = cfg.state_dim_ctrl
        self.action_dim = cfg.action_dim_ctrl
        self.num_heads = 4

        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.clip_ratio = cfg.clip_ratio
        self.K_epochs = cfg.K_epochs
        self.batch_size = cfg.batch_size

        self.entropy_coef = cfg.entropy_coef
        self.value_coef = cfg.value_loss_coef

        self.rollout_size = cfg.n_step  # 1000 = rollout length

        self.buffer = RolloutBuffer()

        self.policy = MultiHeadActorCritic(
            self.state_dim,
            self.action_dim,
            self.num_heads
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=cfg.learning_rate
        )

    # --------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------
    def select_action(self, state):

        state = torch.tensor(state, dtype=torch.float32)\
                    .unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.policy(state)

        actions = []
        log_probs = []

        for head_logits in logits:
            dist = torch.distributions.Categorical(logits=head_logits)
            action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))

        actions = torch.stack(actions, dim=1)      # (1, 4)
        log_probs = torch.stack(log_probs, dim=1)  # (1, 4)

        # Sum log-probs across heads
        total_log_prob = log_probs.sum(dim=1)

        return (
            actions.squeeze(0).cpu().tolist(),
            total_log_prob.item(),
            value.item()
        )

    # --------------------------------------------------
    # STORE TRANSITION
    # --------------------------------------------------
    def remember(self, s, a, logp, v, r, done):
        self.buffer.push(s, a, logp, v, r, done)

    # --------------------------------------------------
    # TRAIN STEP
    # --------------------------------------------------
    def train_step(self):
        if len(self.buffer) >= self.rollout_size:
            self.update()
    
    def compute_gae(self, rewards, values, dones, last_value):

        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):

            if i == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[i+1]

            delta = rewards[i] + \
                    self.gamma * next_value * (1 - dones[i]) - \
                    values[i]

            gae = delta + \
                  self.gamma * self.gae_lambda * \
                  (1 - dones[i]) * gae

            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values)]

        return advantages, returns

    def update(self):
        # Convert rollout to tensors
        states = torch.tensor(self.buffer.states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32).to(self.device)

        rewards = self.buffer.rewards
        dones = self.buffer.is_terminals

        # Bootstrap value
        with torch.no_grad():
            last_state = torch.tensor(
                self.buffer.states[-1],
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            _, last_value = self.policy(last_state)
            last_value = last_value.item()

        advantages, returns = self.compute_gae(
            rewards, values, dones, last_value
        )

        advantages = torch.tensor(advantages,
                                  dtype=torch.float32).to(self.device)

        returns = torch.tensor(returns,
                               dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / \
                     (advantages.std() + 1e-8)

        dataset_size = states.size(0)

        for _ in range(self.K_epochs):

            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.batch_size):

                idx = indices[start:start+self.batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_logp = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                logits, state_values = self.policy(batch_states)
                state_values = state_values.view(-1)

                # Compute new log-prob across heads
                total_logp = 0
                entropy = 0

                for h, head_logits in enumerate(logits):
                    dist = torch.distributions.Categorical(
                        logits=head_logits
                    )
                    total_logp += dist.log_prob(batch_actions[:, h])
                    entropy += dist.entropy().mean()

                ratios = torch.exp(total_logp - batch_old_logp)

                surr1 = ratios * batch_adv
                surr2 = torch.clamp(
                    ratios,
                    1 - self.clip_ratio,
                    1 + self.clip_ratio
                ) * batch_adv

                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * \
                    (state_values - batch_returns).pow(2).mean()

                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 0.5
                )
                self.optimizer.step()

        self.buffer.clear()