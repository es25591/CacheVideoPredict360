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
            self.action_dim
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

        with torch.no_grad():
            _, action_probs = self.network(state)

        action = torch.multinomial(action_probs, 1)

        return action.item()

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        self.learn()

        self.buffer.clear()

    def learn(self):
        batch = self.buffer.sample()
        s, a, r, sn, d = zip(*batch)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(sn).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Compute current values and advantages
        values, log_probs, entropy = self.network(s, a)
        with torch.no_grad():
            next_values, _ = self.network(ns)
            targets = r + self.gamma * next_values * (1 - d)

        advantages = targets - values

        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = self.loss_fn(values, targets)
        entropy_loss = entropy.mean() 

        total_loss = actor_loss + (0.5 * critic_loss) - (self.cfg.entropy_coef * entropy_loss)
    
        # Optimize the network
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.debugger.log('train_loss', total_loss.item())

    def update_epsilon(self):
        self.buffer.clear()
        pass