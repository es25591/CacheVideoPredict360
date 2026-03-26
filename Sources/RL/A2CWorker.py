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
from Sources.RL.Buffers import NStepReplayBuffer, RolloutBuffer
from Sources.RL.Networks import A2CNetwork


class A2CWorker:
    def __init__(self, cfg, debugger=None):
        self.device = resolve_torch_device()
        
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_base_focus
        self.action_dim = cfg.action_dim_base_focus
        self.hidden_dims = cfg.hidden_dims

        self.epsilon = 0.0 
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size

        # self.buffer = RolloutBuffer(cfg.buffer_capacity)
        self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)

        self.network = A2CNetwork(
            state_dim=self.state_dim, 
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        if cfg.optimizer == "adam":
            self.actor_optimizer = optim.Adam(
                self.network.actor.parameters(), lr=cfg.learning_rate_actor
            )
            self.critic_optimizer = optim.Adam(
                self.network.critic.parameters(), lr=cfg.learning_rate_critic
            )
        elif cfg.optimizer == "sgd":
            self.actor_optimizer = optim.SGD(
                self.network.actor.parameters(), lr=cfg.learning_rate_actor
            )
            self.critic_optimizer = optim.SGD(
                self.network.critic.parameters(), lr=cfg.learning_rate_critic
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer,
            gamma=cfg.learning_rate_decay
        )
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer,
            gamma=cfg.learning_rate_decay
        )

        self.scheduler = self.actor_scheduler  # Assuming both schedulers have the same decay

        self.loss_fn = nn.MSELoss()
        self.nb_interval = cfg.nb_interval

    def select_action(self, state):       
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        value, action_probs = self.network(state)

        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), value.squeeze(), log_prob.squeeze(0)

    def remember(self, log_prob, value, reward, next_state, done):
        self.buffer.push(log_prob, value, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        metrics = self.learn()
        
        return metrics

    def learn(self):
        # batch = self.buffer.get_all()
        batch = self.buffer.sample(self.batch_size)

        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Bootstrap the TD target with the critic at next state.
        with torch.no_grad():
            next_value, _ = self.network(next_state)
            next_value = next_value.squeeze()
            target_value = reward + self.gamma * next_value * (1 - done)

        value, log_prob, _ = self.network(state, action)
        advantage = target_value - value.squeeze()

        # Compute losses for each head.
        actor_loss = - (log_prob * advantage.detach()).mean()

        # --- Actor update ---
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.network.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # --- Critic forward pass ---
        value, _ = self.network(state)
        value = value.squeeze()
        critic_loss = self.loss_fn(value, target_value)
        
        # --- Critic update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.network.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        total_loss = actor_loss + critic_loss

        metrics = {
            'train_loss': float(total_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'critic_loss': float(critic_loss.item())
        }

        return metrics

    def update_epsilon(self):
        self.buffer.clear()