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
        self.scheduler.step()
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
