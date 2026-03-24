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
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_base_focus
        self.action_dim = cfg.action_dim_base_focus

        self.epsilon = 0.0 
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size

        self.device = resolve_torch_device()

        self.buffer = RolloutBuffer(cfg.buffer_capacity)

        self.network = A2CNetwork(
            self.state_dim, 
            self.action_dim,
            
        ).to(self.device)

        if cfg.optimizer == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=cfg.learning_rate)
        elif cfg.optimizer == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(), lr=cfg.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=cfg.learning_rate_decay
        )

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
            return

        self.learn()
        self.buffer.clear()

    def learn(self):
        batch = self.buffer.get_all()
        log_prob, value, reward, next_state, done = zip(*batch)

        log_prob = torch.stack(log_prob).to(self.device)
        value = torch.stack(value).to(self.device).squeeze()
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)  

        # Compute current values and advantages
        with torch.no_grad():
            next_value, _ = self.network(next_state)
            next_value = next_value.squeeze()
            target_value = reward + self.gamma * next_value * (1 - done)

        advantage = target_value - value

        # Compute losses
        actor_loss = - (log_prob * advantage.detach()).mean()
        critic_loss = self.loss_fn(value, target_value)
        total_loss = actor_loss + critic_loss
        
        # Optimize the network
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)    
        self.optimizer.step()

        self.debugger.log('train_loss', total_loss.item())

    def update_epsilon(self):
        self.buffer.clear()