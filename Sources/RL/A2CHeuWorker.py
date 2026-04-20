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
from Sources.RL.Buffers import RolloutBuffer
from Sources.RL.Networks import A2CHeuNetwork, A2CNetwork, A2CSharedNetwork
from Sources.RL.ActionRanking import WolpertingerKnnSelector


class LFUHeuristic:
    def __init__(self, cache_size):
        self.cache_size = cache_size * 5

    def get_action(self, state):
        y_l = state[2*self.cache_size + 1]
        x_l = y_l + state[self.cache_size:2*self.cache_size]

        return int(np.argmin(y_l + x_l))

    def get_heuristic_values(self, state):
        cache_size = self.cache_size

        y_s = state[2*cache_size]
        y_l = state[2*cache_size + 1]

        x_s = y_s + state[:cache_size]
        x_l = y_l + state[cache_size:2*cache_size]

        arg_max_s = np.argmin(x_s)
        arg_max_l = np.argmin(x_l)

        return arg_max_s, x_s[arg_max_s], arg_max_l, x_l[arg_max_l]

class A2CHeuWorker:
    def __init__(self, cfg, debugger=None):
        self.device = resolve_torch_device()

        self.cfg = cfg
        self.debugger = debugger

        self.step = 0
        self.n_step = cfg.n_step
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.hidden_dims = cfg.hidden_dims

        self.epsilon = 0.0 
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size

        # A2C-specific hyperparameters
        self.gae_lambda = cfg.gae_lambda
        self.entropy_beta = cfg.entropy_beta
        self.advantage_clip = cfg.advantage_clip
        self.gradient_clip_norm = cfg.gradient_clip_norm

        # A2C is on-policy: keep an ordered rollout and update from contiguous transitions.
        self.buffer = RolloutBuffer(cfg.buffer_capacity)

        self.network = A2CHeuNetwork(
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

        self.teacher = LFUHeuristic(cache_size=cfg.cache_size)

        self.action_call_count = 0
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        _, _, teacher_action, _ = self.teacher.get_heuristic_values(
            state.cpu().numpy().squeeze(0)
        )

        value, logits = self.network(state)
        
        eta = 0.1
        beta = 1.0
        logits[0, teacher_action] += (logits[0].max() - logits[0, teacher_action]) ** beta + eta 

        probs = F.softmax(logits, dim=-1)

        # self.action_call_count += 1
        # if self.action_call_count % 100 == 0:
        #     print("probs =", probs.detach().cpu().numpy())

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        entropy = dist.entropy()

        return action.item(), value.squeeze(), log_prob.squeeze(0), entropy.squeeze(0)

    def train_step(self):
        self.step += 1

        if len(self.buffer) < self.batch_size:# or self.step % self.nb_interval != 0:
            return None

        metrics = self.learn()

        # On-policy update: discard rollout after learning.
        self.buffer.clear()

        return metrics

    def learn(self):
        rollout = list(self.buffer.get_all())
        if not rollout:
            return None

        # Keep the most recent contiguous transitions to preserve temporal structure.
        rollout_horizon = min(len(rollout), self.n_step)
        batch = rollout[-rollout_horizon:]

        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Get values for GAE computation
        with torch.no_grad():
            value, _ = self.network(state)
            value = value.squeeze()
            next_value, _ = self.network(next_state)
            next_value = next_value.squeeze()

        # Convert to numpy for GAE computation on ordered rollout transitions.
        value_np = value.cpu().numpy()
        next_value_np = next_value.cpu().numpy()
        reward_np = reward.cpu().numpy()
        done_np = done.cpu().numpy()

        # Compute GAE (Generalized Advantage Estimation)
        advantages_np, returns_np = self.buffer.compute_gae(
            reward_np, value_np, next_value_np[-1], done_np, gamma=self.gamma, lam=self.gae_lambda
        )

        advantages = torch.FloatTensor(advantages_np).to(self.device)
        returns = torch.FloatTensor(returns_np).to(self.device)

        # # Normalize and clip advantages to stabilize actor updates.
        # advantage_mean = advantages.mean()
        # advantage_std = advantages.std().clamp_min(1e-8)
        # advantages = (advantages - advantage_mean) / advantage_std
        # advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)

        # --- Actor update with entropy regularization ---
        value, logits = self.network(state)
        
        probs = F.log_softmax(logits, dim=-1)
        prob = probs.gather(1, action.unsqueeze(1)).squeeze(1)

        # Actor loss: policy gradient with entropy bonus for exploration
        actor_loss = - (prob * advantages.detach()).mean()
     
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.network.actor.parameters(), max_norm=self.gradient_clip_norm)
        self.actor_optimizer.step()

        # --- Critic update (MSE on TD target) ---
        value, _ = self.network(state)
        value = value.squeeze()
        critic_loss = self.loss_fn(value, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.network.critic.parameters(), max_norm=self.gradient_clip_norm)
        self.critic_optimizer.step()

        total_loss = actor_loss + critic_loss

        metrics = {
            'train_loss': float(total_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'rollout_size': int(rollout_horizon),
            'advantage_min': float(advantages.min().item()),
            'advantage_max': float(advantages.max().item())
        }

        return metrics

    def update_epsilon(self):
        self.buffer.clear()
        self.step = 0
        
    def __str__(self):
        return (
            f"A2CWorker(ActorLR={self.actor_optimizer.param_groups[0]['lr']:.6f}, "
            f"CriticLR={self.critic_optimizer.param_groups[0]['lr']:.6f})\n"
            f"BatchSize={self.batch_size}, Gamma={self.gamma}\n"
            f"BufferSize={self.cfg.buffer_capacity}, GAE_lambda={self.gae_lambda}, Entropy_beta={self.entropy_beta}\n"
            f"Advantage_clip={self.advantage_clip}, Gradient_clip_norm={self.gradient_clip_norm}\n"
            f"hiddens={self.hidden_dims}, Action Dim={self.action_dim}, State Dim={self.state_dim}"
        )
