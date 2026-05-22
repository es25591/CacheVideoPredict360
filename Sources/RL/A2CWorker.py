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
from Sources.RL.Networks import A2CNetwork, A2CSharedNetwork
from Sources.RL.ActionRanking import WolpertingerKnnSelector


class A2CWorker:
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
        self.action_temperature = float(getattr(cfg, "action_temperature", 1.0))
        self.random_action_prob = float(getattr(cfg, "random_action_prob", 0.0))

        self.buffer = RolloutBuffer(cfg.buffer_capacity)
        # self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)

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

        self.scheduler = self.actor_scheduler

        self.loss_fn = nn.MSELoss()
        self.nb_interval = cfg.nb_interval

    def remember(self, s, a, r, ns, d):
        self.buffer.push(s, a, r, ns, d)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        value, probs = self.network(state)

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        entropy = dist.entropy()

        return action.item(), value.squeeze(), log_prob.squeeze(0), entropy.squeeze(0)

    def train_step(self):
        self.step += 1

        if len(self.buffer) < self.batch_size:
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
            val, _ = self.network(state)
            next_value, _ = self.network(next_state)
            val, next_value = val.squeeze(), next_value.squeeze()

        # Convert to numpy for GAE computation on ordered rollout transitions.
        val_np = val.cpu().numpy()
        nxt_val_np = next_value.cpu().numpy()
        reward_np = reward.cpu().numpy()
        done_np = done.cpu().numpy()

        # Compute GAE (Generalized Advantage Estimation)
        advantages_np, returns_np = self.buffer.compute_gae(
            reward_np, val_np, nxt_val_np[-1], done_np, gamma=self.gamma, lam=self.gae_lambda
        )

        advantages = torch.FloatTensor(advantages_np).to(self.device)
        returns = torch.FloatTensor(returns_np).to(self.device)

        # --- Actor update with entropy regularization ---
        val, log_prob, entropy = self.network(state, action)

        # Actor loss: policy gradient with entropy bonus for exploration
        actor_loss = - (log_prob * advantages.detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.network.actor.parameters(), max_norm=self.gradient_clip_norm)
        self.actor_optimizer.step()

        # --- Critic update (MSE on TD target) ---
        val, _ = self.network(state)
        val = val.squeeze()
        critic_loss = self.loss_fn(val, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.network.critic.parameters(), max_norm=self.gradient_clip_norm)
        self.critic_optimizer.step()

        total_loss = actor_loss + critic_loss

        metrics = {
            'train_loss': float(total_loss.item()),
            'actor_loss': float(actor_loss.item()),
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
        
        self.debugger.log('lr', self.scheduler.get_last_lr()[0])
        
    def __str__(self):
        return (
            f"A2CWorker(ActorLR={self.actor_optimizer.param_groups[0]['lr']:.6f}, "
            f"CriticLR={self.critic_optimizer.param_groups[0]['lr']:.6f})\n"
            f"BatchSize={self.batch_size}, Gamma={self.gamma}\n"
            f"BufferSize={self.cfg.buffer_capacity}, GAE_lambda={self.gae_lambda}, Entropy_beta={self.entropy_beta}\n"
            f"Advantage_clip={self.advantage_clip}, Gradient_clip_norm={self.gradient_clip_norm}\n"
            f"hiddens={self.hidden_dims}, Action Dim={self.action_dim}, State Dim={self.state_dim}"
        )

class KnnA2CWorker(A2CWorker):
    """A2C worker with Wolpertinger-like KNN candidate pruning for discrete actions."""

    def __init__(self, cfg, debugger=None):
        super().__init__(cfg, debugger=debugger)

        self.wolpertinger_enabled = bool(getattr(cfg, "wolpertinger_enabled", True))
        base_k = int(getattr(cfg, "wolpertinger_k_base", 8))
        enh_k = int(getattr(cfg, "wolpertinger_k_enh", 4))
        ema_alpha = float(getattr(cfg, "wolpertinger_ema_alpha", 0.05))
        temperature = float(getattr(cfg, "wolpertinger_temperature", 1.2))
        frequency_penalty = float(getattr(cfg, "wolpertinger_frequency_penalty", 0.05))
        random_action_prob = float(getattr(cfg, "wolpertinger_random_action_prob", 0.05))

        self.base_selector = WolpertingerKnnSelector(
            action_dim=self.action_dim,
            embedding_dim=self.hidden_dims[-1],
            k=base_k,
            ema_alpha=ema_alpha,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            random_action_prob=random_action_prob,
        )

    def select_action(self, state, action_space="base"):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        value, action_probs, state_embedding = self.network(state_t, return_embedding=True)

        action_probs = action_probs.squeeze(0)
        action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
        action_probs = action_probs / (action_probs.sum() + 1e-12)

        state_embedding_np = state_embedding.squeeze(0).detach().cpu().numpy()
        probs_np = action_probs.detach().cpu().numpy()

        selector = self.base_selector

        if not self.wolpertinger_enabled or selector.action_dim != probs_np.shape[0]:
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), value.squeeze(), log_prob.squeeze(0)

        action_id, candidates, candidate_probs = selector.select(state_embedding_np, probs_np)

        if candidate_probs.size == 0 or float(candidate_probs.sum()) <= 0.0:
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), value.squeeze(), log_prob.squeeze(0)

        candidate_probs = np.asarray(candidate_probs, dtype=np.float32)
        candidate_probs = candidate_probs / (candidate_probs.sum() + 1e-12)
        chosen_idx = int(
            np.where(candidates == action_id)[0][0]) if action_id in candidates else int(np.argmax(candidate_probs)
        )
        log_prob = torch.tensor(
            np.log(max(candidate_probs[chosen_idx], 1e-12)), dtype=torch.float32, device=self.device
        )

        selector.update(action_id, state_embedding_np)
        return int(action_id), value.squeeze(), log_prob
