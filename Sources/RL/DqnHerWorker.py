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
from Sources.RL.Networks import HierarchicalDQNet, QNetwork, MultiHeadQNetwork, FocusQNetwork, DiscretePDQN


class DqnHerWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.hidden_dims = cfg.hidden_dim

        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        self.device = resolve_torch_device()

        self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)

        self.policy_net = QNetwork(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=self.hidden_dim
        ).to(self.device)
        self.target_net = QNetwork(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=self.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if cfg.optimizer == "adam":
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        elif cfg.optimizer == "sgd":
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=cfg.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=cfg.learning_rate_decay
        )

        self.loss_fn = nn.MSELoss()
        self.nb_interval = cfg.nb_interval

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            qvals = self.policy_net(state_t)

        return qvals.argmax().item()

    def remember(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def train_step(self):
        self.step += 1
        if len(self.buffer) >= self.batch_size and self.step % self.nb_interval == 0:
            self.learn()

    def learn(self):
        batch = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s = torch.tensor(np.stack(s), dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.stack(ns), dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.int64).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            n_step_gamma = self.gamma ** self.n_step
            q_target = r + n_step_gamma * next_q * (1.0 - d)

        q_expected = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.update_target()

        self.debugger.log('train_loss_base', loss.item())
        self.debugger.log('epsilon_base', self.epsilon)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        for tparam, pparam in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tparam.data.copy_(tparam.data * (1.0 - self.tau) + pparam.data * self.tau)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step = 0

    def __str__(self):
        return (
            f"DQNWorker(step={self.step}, epsilon={self.epsilon:.4f}, "
            f"buffer_size={len(self.buffer)}, learning_rate={self.optimizer.param_groups[0]['lr']:.6f}, gamma={self.gamma}, tau={self.tau}\n"
            f"policy_net_params={sum(p.numel() for p in self.policy_net.parameters())}, "
            f"target_net_params={sum(p.numel() for p in self.target_net.parameters())}), "
            f"optimizer={self.optimizer.__class__.__name__}\n"
            f"hiddens={self.hidden_dims}, Action Dim={self.action_dim}, State Dim={self.state_dim}"
        )

