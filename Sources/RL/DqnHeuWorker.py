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


class CacheEvictionHeuristic:
    def __init__(self, num_rows, num_cols, cache_size):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cache_size = cache_size
        self.action_dim = cache_size
        
        # Pre-compute the 2D grid coordinates for every possible tile ID
        total_tiles = num_rows * num_cols
        self.tile_coords = {
            i: (i // num_cols, i % num_cols) for i in range(total_tiles)
        }

    def compute_heuristic(self, current_viewed_tile, cache_state):
        """
        Generates the H(s, a) array for cache replacement.
        current_viewed_tile: The tile ID the user is currently looking at.
        cache_state: A list or array of size `cache_size`. 
                     Contains the tile IDs currently in the cache. 
                     Use -1 to represent an empty slot.
        Returns: A numpy array of shape [cache_size]
        """
        h_values = np.zeros(self.action_dim, dtype=np.float32)

        target_row, target_col = self.tile_coords[current_viewed_tile]

        for slot_index in range(self.cache_size):
            cached_tile_id = cache_state[slot_index]
            
            # Rule 1: Always prioritize filling empty slots first
            if cached_tile_id == -1:
                h_values[slot_index] = 1.0
                continue
                
            # Extract coordinates of the tile currently in this cache slot
            row, col = self.tile_coords[cached_tile_id]
            
            # Calculate cyclic yaw distance
            col_dist = min(abs(col - target_col), self.num_cols - abs(col - target_col))
            # Calculate pitch distance
            row_dist = abs(row - target_row)
            
            # Squared Euclidean distance
            distance_sq = (row_dist**2) + (col_dist**2)
            
            # Rule 2: Inverted Proximity. 
            # If distance is 0 (tile is in viewport), proximity is 1 -> H becomes 0 (DO NOT EVICT).
            # If distance is large, proximity approaches 0 -> H approaches 1 (EVICT THIS).
            proximity = 1.0 / (1.0 + distance_sq)
            h_values[slot_index] = 1.0 - proximity

        return h_values

class LFUHeuristic:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        
    def get_action(self, state):
        state = np.array(state)

        # Add small noise to avoid deterministic ties
        noisy_state = state

        # LFU: evict the item with minimum frequency
        return int(np.argmin(noisy_state))

    def get_heuristic_values(self, state):
        # Deconcatenate state: [x_s, x_l, y_s, y_l, step_indicators]
        cache_size = self.cache_size
        y_s = state[2*cache_size]
        y_l = state[2*cache_size + 1]
        
        x_s = y_s + state[:cache_size]
        x_l = y_l + state[cache_size:2*cache_size]
        
        # step_indicators are not needed for LFU heuristic
        
        # Invert frequencies: lower freq → higher heuristic value
        # Focus on cache slot frequencies, not current request
        
        max_freq_s = np.max(x_s)
        max_freq_l = np.max(x_l)
        
        h_cache_s = max_freq_s - x_s
        h_cache_l = max_freq_l - x_l
        
        alpha = 0.5
        h = alpha * h_cache_s + (1 - alpha) * h_cache_l

        return h

class DqnHeuWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.hidden_dim = cfg.hidden_dim

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

        self.teacher = LFUHeuristic(
            cache_size=cfg.action_dim
        )
        self.recent_actions = []
        self.omega = cfg.omega

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            qvals = self.policy_net(state_t)

        print("Q-values before heuristic adjustment:", np.sum(qvals.cpu().numpy()))

        heuristic_values = self.teacher.get_heuristic_values(state)

        h_tensor = torch.tensor(heuristic_values, dtype=torch.float32).to(self.device)        
        qvals = qvals + (self.omega * h_tensor)

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
        # self.scheduler.step()

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
            f"hiddens={self.hidden_dim}, Action Dim={self.action_dim}, State Dim={self.state_dim}"
        )

