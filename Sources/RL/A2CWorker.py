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
from Sources.RL.Networks import A2CNetwork
from Sources.RL.ActionRanking import WolpertingerKnnSelector


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
        
        # A2C-specific hyperparameters
        self.gae_lambda = cfg.gae_lambda
        self.entropy_beta = cfg.entropy_beta
        self.advantage_clip = cfg.advantage_clip
        self.gradient_clip_norm = cfg.gradient_clip_norm

        # A2C is on-policy: keep an ordered rollout and update from contiguous transitions.
        self.buffer = RolloutBuffer(cfg.buffer_capacity)

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

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def select_action(self, state, action_space="base"):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        value, action_probs = self.network(state)

        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), value.squeeze(), log_prob.squeeze(0)

    def train_step(self):
        self.step += 1
        if len(self.buffer) < self.batch_size or self.step % self.nb_interval != 0:
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

        # Normalize advantages to reduce variance and improve learning stability
        # (helps with PSNR reward scale 0-40)
        advantage_mean = advantages.mean()
        advantage_std = advantages.std() + 1e-8  # Add small epsilon to avoid division by zero
        advantages_1 = (advantages - advantage_mean) / advantage_std

        # print(f"Advantages: {advantages:.4f}")
        # print(f"Advantages before normalization: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}")
        # print(f"Advantages normalized: mean={advantages_1.mean().item():.4f}, std={advantages_1.std().item():.4f}")
        # print(f"Advantages before clipping: min={advantages.min().item():.4f}, max={advantages.max().item():.4f}")
        
        # # Optional: Clip advantages to prevent extreme policy updates
        # advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)

        # --- Actor update with entropy regularization ---
        value, log_prob, entropy = self.network(state, action)

        # Actor loss: policy gradient with entropy bonus for exploration
        actor_loss = - (log_prob * advantages.detach()).mean()
        entropy_loss = -self.entropy_beta * entropy.mean()  # Negative because we want to maximize entropy
        actor_total_loss = actor_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
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

        total_loss = actor_total_loss + critic_loss

        metrics = {
            'train_loss': float(total_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'entropy_loss': float(entropy_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'policy_entropy': float(entropy.mean().item()),
            'rollout_size': int(rollout_horizon),
            'advantage_min': float(advantages.min().item()),
            'advantage_max': float(advantages.max().item())
        }

        return metrics

    def update_epsilon(self):
        self.buffer.clear()
        self.step = 0


class KnnA2CWorker(A2CWorker):
    """A2C worker with Wolpertinger-like KNN candidate pruning for discrete actions."""

    def __init__(self, cfg, debugger=None):
        super().__init__(cfg, debugger=debugger)

        self.wolpertinger_enabled = bool(getattr(cfg, "wolpertinger_enabled", True))
        base_k = int(getattr(cfg, "wolpertinger_k_base", 8))
        enh_k = int(getattr(cfg, "wolpertinger_k_enh", 4))
        ema_alpha = float(getattr(cfg, "wolpertinger_ema_alpha", 0.1))

        self.base_selector = WolpertingerKnnSelector(
            action_dim=self.action_dim,
            k=base_k,
            ema_alpha=ema_alpha,
        )
        self.enh_selector = WolpertingerKnnSelector(
            action_dim=int(getattr(cfg, "action_dim_enh_focus", self.action_dim)),
            k=enh_k,
            ema_alpha=ema_alpha,
        )

    def _selector_for_space(self, action_space: str) -> WolpertingerKnnSelector:
        if action_space == "enh":
            return self.enh_selector
        return self.base_selector

    def select_action(self, state, action_space="base"):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        value, action_probs = self.network(state_t)
        probs_np = action_probs.squeeze(0).detach().cpu().numpy()

        # If action-space dimensions mismatch network output, fallback to vanilla sampling.
        selector = self._selector_for_space(action_space)
        if (not self.wolpertinger_enabled) or selector.action_dim != probs_np.shape[0]:
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), value.squeeze(), log_prob.squeeze(0)

        action_id, candidates = selector.select(probs_np)
        probs_safe = np.nan_to_num(probs_np, nan=0.0, posinf=0.0, neginf=0.0)
        candidate_mass = float(probs_safe[candidates].sum())

        if candidate_mass <= 0.0:
            # Use full distribution if candidate distribution is degenerate.
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), value.squeeze(), log_prob.squeeze(0)

        action_tensor = torch.tensor(action_id, device=self.device)
        log_prob_value = np.log(max(probs_safe[action_id] / candidate_mass, 1e-12))
        log_prob = torch.tensor(log_prob_value, dtype=torch.float32, device=self.device)

        return int(action_id), value.squeeze(), log_prob