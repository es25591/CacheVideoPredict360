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


class FocusWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_base_focus
        self.action_dim_base = cfg.action_dim_base_focus
        self.action_dim_enh = cfg.action_dim_enh_focus
        
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        self.device = resolve_torch_device()

        self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)

        hidden_dims_focus = cfg.hidden_dims_focus

        self.policy_net = FocusQNetwork(
            self.state_dim, 
            self.action_dim_base, 
            self.action_dim_enh, 
            hidden_dims=hidden_dims_focus
        ).to(self.device)
        
        self.target_net = FocusQNetwork(
            self.state_dim, 
            self.action_dim_base, 
            self.action_dim_enh, 
            hidden_dims=hidden_dims_focus
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
        
    # def select_action(self, state):
    #     # Epsilon-greedy exploration
    #     if random.random() < self.epsilon:
    #         a_base = random.randint(0, self.action_dim_base - 1)
    #         a_enh = [random.randint(0, self.action_dim_enh - 1) for _ in range(4)]
    #         return [a_base] + a_enh  # Returns list: [base, enh1, enh2, enh3, enh4]

    #     # Exploitation
    #     state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         q_base, q_enh = self.policy_net(state_t)
            
    #         a_base = q_base.argmax(dim=1).item()
    #         a_enh = q_enh.argmax(dim=2).squeeze(0).tolist()
            
    #         return [a_base] + a_enh

    def select_action(self, state, video_cache_idx):

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:
            a_base = random.randint(0, self.action_dim_base - 1)
        else:
            with torch.no_grad():
                q_base, _ = self.policy_net(state_t) 
            a_base = torch.argmax(q_base, dim=1).item()

        is_video_available = (video_cache_idx != -1) or (a_base != 0)
        
        # 3. Prepare the Enhancement Mask based on that decision
        enh_mask = torch.ones((1, 4, self.action_dim_enh), dtype=torch.bool)
        if not is_video_available:
            enh_mask[:, :, 1:] = False 
        
        # 5. Select Enhancement Actions
        if random.random() < self.epsilon:
            a_enh = []
            for h in range(4):
                valid_indices = torch.where(enh_mask[0, h])[0].tolist()
                a_enh.append(random.choice(valid_indices))
        else:
            with torch.no_grad():
                _, q_enh = self.policy_net(state_t, enh_mask=enh_mask)
            a_enh = torch.argmax(q_enh, dim=2).squeeze().tolist()
            
        return [a_base] + a_enh

    def remember(self, s, action, r, ns, done, is_cached):
        self.buffer.push(s, action, r, ns, done, is_cached)

    def train_step(self):
        self.step += 1
        if self.step % self.nb_interval == 0 and len(self.buffer) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = self.buffer.sample(self.batch_size)
        s, a, r, ns, d, is_cached = zip(*batch)

        s = torch.tensor(np.stack(s), dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.stack(ns), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.stack(a), dtype=torch.long).to(self.device) # Shape: (batch, 5)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)
        is_cached = torch.tensor(is_cached, dtype=torch.bool).to(self.device)

        # Split actions back into Base and Enh
        a_base = a[:, 0]
        a_enh = a[:, 1:]

        # ---------- Calculate Target ----------
        with torch.no_grad():
            next_q_base, next_q_enh = self.target_net(ns)
            
            q_max_next_base = next_q_base.max(1)[0]
            q_max_next_enh = next_q_enh.max(dim=2)[0] # Shape: (batch, 4)
            
            n_step_gamma = self.gamma ** self.n_step
            
            # Base and Enh targets
            q_target_base = r + n_step_gamma * q_max_next_base * (1.0 - d)
            
            # Need to unsqueeze r and d to broadcast properly against the 4 Enh heads
            q_target_enh = r.unsqueeze(1) + n_step_gamma * q_max_next_enh * (1.0 - d).unsqueeze(1)

        # ---------- Calculate Expected ----------
        q_base, _ = self.policy_net(s)
        _, q_enh = self.policy_net(s, enh_mask=is_cached)

        q_expected_base = q_base.gather(1, a_base.unsqueeze(1)).squeeze(1)
        q_expected_enh = q_enh.gather(2, a_enh.unsqueeze(2)).squeeze(2)

        # ---------- Combined Loss ----------
        loss_base = self.loss_fn(q_expected_base[~is_cached], q_target_base[~is_cached])
        loss_enh = self.loss_fn(q_expected_enh, q_target_enh)

        loss = loss_base + loss_enh

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.update_target()

        # Logging
        if self.debugger:
            self.debugger.log('train_loss_combined', loss.item())
            self.debugger.log('train_loss_base', loss_base.item())
            self.debugger.log('train_loss_enh', loss_enh.item())
            self.debugger.log('epsilon_combined', self.epsilon)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        for tparam, pparam in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tparam.data.copy_(tparam.data * (1.0 - self.tau) + pparam.data * self.tau)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_step(self):
        self.step = 0

class PDQNWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim
        
        # Now requires both action dimensions
        self.action_dim_base = cfg.action_dim
        self.action_dim_enh = cfg.action_dim

        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        self.device = resolve_torch_device()

        self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)

        # Initialize the P-DQN architecture
        self.policy_net = HierarchicalDQNet(
            self.state_dim, 
            self.action_dim_base, 
            self.action_dim_enh, 
            hidden_dim=cfg.hidden_dim_base_focus
        ).to(self.device)
        
        self.target_net = HierarchicalDQNet(
            self.state_dim, 
            self.action_dim_base, 
            self.action_dim_enh, 
            hidden_dim=cfg.hidden_dim_base_focus
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
            a_base = random.randint(0, self.action_dim_base - 1)
            a_enh = [random.randint(0, self.action_dim_enh - 1) for _ in range(4)]
            return [a_base] + a_enh

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            
            # The network returns the base Q-values, enh Q-values, and the greedily chosen enhancements
            _, _, chosen_base, chosen_enh = self.policy_net(state_t)
            
            a_base = chosen_base.item()
            a_enh = chosen_enh.squeeze(0).tolist()
            
            return [a_base] + a_enh

    def remember(self, s, action, r, ns, done):
        # 'action' is expected to be a list of 5 elements
        self.buffer.push(s, action, r, ns, done)

    def train_step(self):
        self.step += 1
        if self.step % self.nb_interval == 0 and len(self.buffer) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s = torch.tensor(np.stack(s), dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.stack(ns), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.stack(a), dtype=torch.int64).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        # Split actions
        a_base = a[:, 0]
        a_enh = a[:, 1:]
        # print(f"Batch shapes - s: {s.shape}, ns: {ns.shape}, a_base: {a_base.shape}, a_enh: {a_enh.shape}, r: {r.shape}, d: {d.shape}")
                
        # ---------- Target ----------
        with torch.no_grad():
            # Pass next state. Without 'chosen_enh_actions', it greedily picks the best next parameters
            next_q_base, next_q_enh, _, _ = self.target_net(ns)
            
            q_max_next_base = next_q_base.max(1)[0]
            q_max_next_enh = next_q_enh.max(dim=2)[0]
            
            n_step_gamma = self.gamma ** self.n_step

            q_target_base = r + n_step_gamma * q_max_next_base * (1.0 - d)
            q_target_enh = r.unsqueeze(1) + n_step_gamma * q_max_next_enh * (1.0 - d).unsqueeze(1)

            # Apply mask to target (forces target to 0 if base action is 0)
            mask = (q_max_next_base > 0).float()
            q_target_enh = mask.unsqueeze(1) * q_target_enh   # Shape: [Batch, 4]

            # print(f"Target shapes - q_target_base: {q_target_base.shape}, q_target_enh: {q_target_enh.shape}")

        # ---------- Expected ----------
        # Crucially, pass the ACTUAL enhancements taken into the policy net to condition the Base Q-value
        q_base, q_enh, _, _ = self.policy_net(s)

        q_expected_base = q_base.gather(1, a_base.unsqueeze(1)).squeeze(1)
        q_expected_enh = q_enh.gather(2, a_enh.unsqueeze(2)).squeeze(2)

        mask = (q_expected_base > 0).float()
        q_expected_enh = mask.unsqueeze(1) * q_expected_enh   # Shape: [Batch, 4]

        # ---------- Loss Calculation (L) ----------
        
        # Base Loss: L_base = E[(y^0 - Q^0)^2]
        loss_base = self.loss_fn(q_expected_base, q_target_base)

        # Enhancement Loss: L_enh = E[ 1/4 * sum_tau( m_t * (y^tau - Q^tau)^2 ) ]
        # 1. Compute squared error element-wise
        errors_enh = (q_expected_enh - q_target_enh) ** 2       # Shape: [Batch, 4]
        
        # 3. Compute mean (implements the 1/4 sum and Expectation over batch)
        loss_enh = errors_enh.mean()

        # Total Loss
        loss = loss_base + loss_enh

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        self.update_target()

        if self.debugger:
            self.debugger.log('train_loss_pdqn', loss.item())
            self.debugger.log('train_loss_base', loss_base.item())
            self.debugger.log('train_loss_enh', loss_enh.item())
            self.debugger.log('epsilon', self.epsilon)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        for tparam, pparam in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tparam.data.copy_(tparam.data * (1.0 - self.tau) + pparam.data * self.tau)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class BaseWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_base_focus
        self.action_dim = cfg.action_dim # cfg.action_dim_base_focus

        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        self.device = resolve_torch_device()

        self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)

        self.policy_net = QNetwork(self.state_dim, self.action_dim, hidden_dim=cfg.hidden_dim_base_focus).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden_dim=cfg.hidden_dim_base_focus).to(self.device)
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
        if self.step % self.nb_interval == 0 and \
           len(self.buffer) >= self.batch_size:
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


class EnhWorker:
    def __init__(self, cfg, debugger=None):
        self.cfg = cfg
        self.debugger = debugger
        
        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim_enh_focus
        self.action_dim = cfg.action_dim # cfg.action_dim_enh_focus

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
            num_heads=4,
            hidden_dim=cfg.hidden_dim_enh_focus
        ).to(self.device)
        self.target_net = MultiHeadQNetwork(
            self.state_dim,
            self.action_dim,
            num_heads=4,
            hidden_dim=cfg.hidden_dim_enh_focus
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
