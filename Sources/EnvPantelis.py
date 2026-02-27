#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_optimizer as optim_extra

from time import sleep
from collections import deque, defaultdict
from itertools import count
from typing import Any, Dict, Counter, List
from dataclasses import dataclass

sources_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if sources_path not in sys.path:
    sys.path.append(sources_path)

from importnb import Notebook
with Notebook():
    from Labs.CacheEngine import CacheEngineEnv
    from Labs.EnvWrapperPantelis import EnvWrapper
    from Labs.LatencyModel import LatencyModel, MultiDULatencyModel
    from Labs.UserRequest import UserRequestEvents

from RL.Networks import QNetwork, MultiHeadQNetwork
# from RL.Buffers import ReplayBuffer, NStepReplayBuffer
from RL.Adapters import FeatureAdapter

import Common.config as config
import Common.datatypes as datatypes
import Common.debugger as debugger
import Common.utils as utils

importlib.reload(config)
importlib.reload(datatypes)
importlib.reload(debugger)
importlib.reload(utils)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UserTransition = datatypes.UserTransition
CachePolicy = datatypes.CachePolicy
CacheKey = datatypes.CacheKey

cfg = config.Config()
cfg.action_dim = 5 * cfg.cache_size + 1
cfg.filename = \
    f"pan_eps{cfg.epsilon_start}_" \
    f"lrdecay{cfg.learning_rate_decay}_" \
    f"gamma{cfg.gamma}.csv"


debugger = debugger.debug


# In[ ]:


class DrlPolicy(CachePolicy):
    def __init__(self, cfg: Any = None):
        self.cfg = cfg
        self.cur_size = 0

        self.video_idx = [-1] * self.cfg.cache_size
        self.tile_idx = [[-1] * self.cfg.viewport for _ in range(self.cfg.cache_size)]

    def get(self, key: CacheKey) -> Any:
        return self.cache.get(key, None)
    
    def put(self, vid_slot: int, value: Any, size: int) -> list:
        """
        vid_slot   -> slot index
        value -> (video_id, tiles)
        size  -> fixed as 1 slot
        """
        evicted = []
        new_video, _ = value
        
        if new_video in self.video_idx:
            self.cur_size = sum(1 for v in self.video_idx if v != -1)
            return evicted

        self.video_idx[vid_slot] = new_video
        self.tile_idx[vid_slot] = [-1] * self.cfg.viewport

        self.cur_size = sum(1 for v in self.video_idx if v != -1)
        return evicted

    def contains(self, vid: int) -> bool:
        return vid in self.video_idx

    def remove(self, vid: int) -> bool:
        if vid in self.video_idx:
            idx = self.video_idx.index(vid)
            self.video_idx[idx] = -1
            self.tile_idx[idx] = [-1] * self.cfg.viewport
            self.cur_size = sum(1 for v in self.video_idx if v != -1)
            return True
        return False

    def clear(self) -> None:
        self.video_idx = [-1] * self.cfg.cache_size
        self.tile_idx = [[-1] * self.cfg.viewport for _ in range(self.cfg.cache_size)]
        self.cur_size = 0

    def keys(self):
        return self.video_idx
    
    def get_capacity(self) -> int:
        return self.cur_size
    
    def update_size(self):
        self.cur_size = sum(1 for v in self.video_idx if v != -1)

    def stats(self) -> Dict[str, Any]:
        return {
            'current_size': self.cur_size,
            'capacity': self.cfg.cache_size,
            'num_items': len([v for v in self.video_idx if v != -1])
        }


# In[ ]:


class NStepReplayBuffer:
    def __init__(self, capacity, n_step, gamma):
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, s, a, r, ns, done):
        self.n_step_buffer.append((s, a, r, ns, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Compute N-step discounted reward
        # G = r1 + gamma*r2 + ... + gamma^(n-1)*rn
        reward, next_state, done_ = self._get_n_step_info()
        state, action, _, next_state, _ = self.n_step_buffer[0]
        self.memory.append((state, action, reward, next_state, done_))

    # Pantelis - This function computes the normalized n-step reward and returns the next state and done flag from the last transition in the n-step buffer.
    def _get_n_step_info(self):
        reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            reward += transition[2]

        reward = reward / self.n_step # Normalize reward by n_step to prevent large values

        return reward, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.step = 0
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim # 5 * cfg.cache_size + 1
        self.n_step = cfg.n_step

        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        
        self.buffer = NStepReplayBuffer(cfg.buffer_capacity, self.n_step, self.gamma)
        
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=cfg.learning_rate_decay
        )
        
        self.loss_fn = nn.MSELoss()
        self.nb_interval = cfg.nb_interval

    def select_action(self, state, j, idx):
        
        if random.random() < self.epsilon:
            if j == 0:
                return random.randint(0, self.cfg.cache_size), None
            else:
                offset = self.cfg.cache_size + idx * 4 + 1
                action = random.randint(offset, offset + 4)

                return 0 if action == offset + 4 else action, None

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)

        if j == 0:
            action_idx = q_values[0, 0:self.cfg.cache_size].argmax().item()
            return action_idx, q_values
        else:
            offset = self.cfg.cache_size + idx * 4 + 1
            slice_vals = q_values[0, offset : offset + 4]
            max_slice, max_idx = slice_vals.max(0)

            action = 0 if q_values[0, 0] >= max_slice else (offset + max_idx.item())

            return action, q_values

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

        s = torch.tensor(np.stack(s), dtype=torch.float32).to(device)
        ns = torch.tensor(np.stack(ns), dtype=torch.float32).to(device)
        a = torch.tensor(a, dtype=torch.int64).to(device)
        r = torch.tensor(r, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).to(device)

        with torch.no_grad():
            q_next = self.target_net(ns).max(1)[0]
            q_target = r + self.gamma * q_next * (1.0 - d)

        q_expected = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.update_target()

        debugger.log('train_loss', loss.item())
        debugger.log('epsilon', self.epsilon)

    def soft_update_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_( 
                target_param.data * (1.0 - self.tau) + policy_param.data * self.tau
            )

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def reset_step(self):
        self.step = 0


# In[ ]:


class NetworkAdapter:
    def __init__(self, env: Any, feature_adapter: Any, cfg: Any):
        self.env = env
        self.cfg = cfg
        self.features = feature_adapter

        self.C = self.cfg.cache_size  # paper's cache capacity (videos)
        self.k = self.cfg.viewport    # paper's tiles per video (enhancement)

    def build_observation(self, vid: int, tile: int = None) -> np.ndarray:
        video_cache_index = self.env.mec_cache.policy.video_idx
        tile_cache_index = self.env.mec_cache.policy.tile_idx

        x_s = np.zeros(self.C, dtype=np.float32)
        x_l = np.zeros(self.C, dtype=np.float32)

        y_s = np.zeros(self.C * self.k, dtype=np.float32)
        y_l = np.zeros(self.C * self.k, dtype=np.float32)

        for vid_i, v in enumerate(video_cache_index):
            if v == -1:
                continue

            x_s[vid_i] = self.features.video_freq_short.get(v, 0)
            x_l[vid_i] = self.features.video_freq_long.get(v, 0)

            tiles = tile_cache_index[vid_i]

            for til_i, t in enumerate(tiles):
                if t == -1:
                    continue
                y_s[vid_i * self.k + til_i] = self.features.tile_freq_short.get((v, t), 0)
                y_l[vid_i * self.k + til_i] = self.features.tile_freq_long.get((v, t), 0)

        if tile is None:
            z_s = np.array(
                [self.features.video_freq_short.get(vid, 0)], dtype=np.float32
            )
            z_l = np.array(
                [self.features.video_freq_long.get(vid, 0)], dtype=np.float32
            )
        else:
            z_s = np.array(
                [self.features.tile_freq_short.get((vid, tile), 0)], dtype=np.float32
            )
            z_l = np.array(
                [self.features.tile_freq_long.get((vid, tile), 0)], dtype=np.float32
            )

        features = np.concatenate([x_s, x_l, y_s, y_l, z_s, z_l], axis=0)
        features = np.log1p(features)  # Log-transform to reduce scale and handle zeros

        return features

    def reset(self):
        
        obs, info = self.env.reset()
        self.features.reset_history()

        return obs, info
    
    def env_is_done(self) -> bool:
        return self.env.users_env.all_users_done()


# In[ ]:


def save_training_results(
    path_,
    filename,
    ep, 
    total_reward, 
    cache_hits, 
    cache_misses, 
    agent
):
    with open(os.path.join(path_, filename), 'a', newline='') as f:
        fieldnames = [
            'episode', 
            'total_reward', 
            'cache_hits', 
            'cache_misses', 
            'epsilon',
            'lr'
        ]
        writer_results = csv.DictWriter(f, fieldnames=fieldnames)

        if ep == 0:
            writer_results.writeheader()
        
        writer_results.writerow({
            'episode': ep,
            'total_reward': round(float(total_reward), 2),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'lr': f"{agent.scheduler.get_last_lr()[0]:.10f}" if agent else None,
            'epsilon': round(float(agent.epsilon), 4) if agent else None
        })

def update_metrics(info: dict, reward: float) -> tuple[float, int, int, float, float]:
    enh_hits = info.get("enh_layer_hits", 0)
    base_hits = info.get("base_layer_hits", 0)
    enh_misses = info.get("enh_layer_misses", 0)
    base_misses = info.get("base_layer_misses", 0)

    return reward, base_hits, base_misses, enh_hits, enh_misses

def build_latency_model(cfg):
    """Build and return the MultiDULatencyModel."""
    P = cfg.n_nodes
    max_U = cfg.n_users

    return MultiDULatencyModel(
        P=P,
        max_U=max_U,
        R_M_D=80e6,
        R_C_M=125e6,
        mu=2e7,
        eta=2e5,
        B_pu_matrix=np.full((P, max_U), 20e6, dtype=float),
        gamma_pu_matrix=np.full((P, max_U), 5.0, dtype=float),
        rhoT_p=[0.2],
        lambda_p=[0.05],
        du_fixed_delay=0.001,
        mec_fixed_delay=0.005,
        cloud_fixed_delay=0.1
    )

def build_environment(cfg):
    """Construct the full multi-component environment wrapper."""
    du_caches = []

    # DRL Caching Policy
    policy = DrlPolicy(cfg=cfg)

    # MEC Cache Engine
    mec_cache = CacheEngineEnv(
        n_users=cfg.n_users,
        n_videos=cfg.n_videos,
        n_layers=cfg.n_layers,
        n_tiles=cfg.n_tiles,
        n_gops=cfg.n_gops,
        cache_capacity=cfg.cache_capacity,
        policy=policy
    )

    # User request generator
    users_env = UserRequestEvents(
        n_nodes=cfg.n_nodes,
        n_users=cfg.n_users,
        n_videos=cfg.n_videos,
        n_gops=cfg.n_gops,
        n_layers=cfg.n_layers,
        n_tiles=cfg.n_tiles,
        n=cfg.n,
        m=cfg.m,
        arrival_rate=cfg.arrival_rate,
        zipf_alpha=cfg.zipf_alpha
    )

    # Latency Model
    latency_model = build_latency_model(cfg)

    # Wrapping all into the main training environment
    return EnvWrapper(
        cfg=cfg,
        n=cfg.n,
        m=cfg.m,
        n_layers=cfg.n_layers,
        users_env=users_env,
        du_caches=du_caches,
        mec_cache=mec_cache,
        latency_model=latency_model,
        theta=cfg.theta,
        lam=cfg.lam,
        max_steps=cfg.max_steps,
        prefetch_fn=lambda cache, action: cache.drl_prefetching_pantelis(action),
        reward_fn=lambda env, reqs: env.compute_reward(reqs),
        debugger=debugger
    )


# In[ ]:


def run_episode(episode, env, agent, net_adapter, cfg):
    _, info = net_adapter.reset()
    agent.reset_step()

    total_reward = 0.0
    cache_hits = cache_misses = 0
    base_hits = base_misses = 0
    enh_hits = enh_misses = 0

    env.warmup_phase(net_adapter)

    for step in count():
        req_state = info["user_request"]

        _, reward, _, info = env.step(
            agent=agent, 
            net_adapter=net_adapter, 
            req=req_state, 
            cfg=cfg
        )

        delta_r, bs_hits, bs_misses, e_hits, e_misses = update_metrics(info, reward)
        total_reward += delta_r
        cache_hits += bs_hits + e_hits
        cache_misses += bs_misses + e_misses
        base_hits += bs_hits
        base_misses += bs_misses
        enh_hits += e_hits
        enh_misses += e_misses

        if net_adapter.env_is_done():
            break

        debugger.log('step_reward', reward)
        debugger.log('cumulative_reward', total_reward)
        debugger.log('cache_hits', cache_hits)
        debugger.log('cache_misses', cache_misses)

    return total_reward, cache_hits, cache_misses, base_hits, base_misses, enh_hits, enh_misses

def train(cfg):
    env = build_environment(cfg)
    agent = DQNAgent(cfg)

    feature_adapter = FeatureAdapter(env, cfg)
    net_adapter = NetworkAdapter(env, feature_adapter, cfg)
    
    date_dir = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")

    print(f"Starting training for {cfg.n_episodes} episodes... {date_dir}")

    for episode in range(cfg.n_episodes):

        total_reward, hits, misses, bs_hits, bs_miss, enh_hits, enh_miss = run_episode(
            episode, env, agent, net_adapter, cfg
        )

        agent.update_epsilon()

        save_training_results(
            path_=cfg.path_results,
            filename=cfg.filename,
            ep=episode,
            total_reward=total_reward,
            cache_hits=hits,
            cache_misses=misses,
            agent=agent
        )

        print(
            f"--- Episode {episode} | R: {total_reward:.2f} | "
            f"HR: {hits / (hits + misses + 1e-9):.2f} | "
            f"BHR: {bs_hits / (bs_hits + bs_miss + 1e-9):.2f} | "
            f"EHR: {enh_hits / (enh_hits + enh_miss + 1e-9):.2f} ---"
        )
        
        debug_path = os.path.join(cfg.path_results, date_dir)
        os.makedirs(debug_path, exist_ok=True)

        # debugger.histogram("base_layer", "Base Controller Decisions")
        # debugger.histogram("enh_layer_1", "Enh Layer 1 Controller Decisions")
        # debugger.histogram("enh_layer_2", "Enh Layer 2 Controller Decisions")
        # debugger.histogram("enh_layer_3", "Enh Layer 3 Controller Decisions")
        # debugger.histogram("enh_layer_4", "Enh Layer 4 Controller Decisions")

        debugger.save_results(filepath=f"{debug_path}/debug_ep{episode}")
        debugger.clear()

        print("-" * 50)

if __name__ == "__main__":
    train(cfg)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------      
# 1. Publication Style Configuration
# ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True
})

# Professional color for line plots
PRIMARY_COLOR = "#2b7bba"

# ----------------------------------------------------------------
# 2. Data Loading & Smoothing
# ----------------------------------------------------------------
cfg.filename = "drl_pan_eps0.987_lrdecay0.9999_c50_ar200.0_z0.8.csv"
path = cfg.path_results + "/ceres/" + cfg.filename

print(f"Loading data from: {path}")

df = pd.read_csv(path)

total = (df["cache_hits"] + df["cache_misses"]).replace(0, np.nan)
df["hit_rate"] = (df["cache_hits"] / total) * 100
df["miss_rate"] = (df["cache_misses"] / total) * 100

# Metrics to plot
metrics = ["total_reward", "hit_rate", "miss_rate", "epsilon", "lr"]
window_size = 1  # Adjust smoothing window as needed

# ----------------------------------------------------------------
# 3. Plotting logic
# ----------------------------------------------------------------
# Adjusted figsize for a 4-column row (standard for full-width paper figures)
fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3), sharex=True)

for ax, col in zip(axes, metrics):
    # Plot raw data with transparency (alpha)
    ax.plot(df["episode"], df[col], color=PRIMARY_COLOR, alpha=0.3, linewidth=0.8, label='Raw')
    
    # Plot moving average for clearer trend (except for epsilon which is usually linear)
    if col != "epsilon":
        smoothed = df[col].rolling(window=window_size).mean()
        ax.plot(df["episode"], smoothed, color=PRIMARY_COLOR, linewidth=1.5, label='Trend')
    else:
        # Just a solid line for Epsilon
        ax.plot(df["episode"], df[col], color=PRIMARY_COLOR, linewidth=1.5)

    # Stylistic cleanup
    ax.set_title(col.replace("_", " ").title(), fontweight="bold")
    ax.set_xlabel("Episode")
    
    # Remove redundant Y-labels to save space, or keep for clarity
    ax.set_ylabel("Value") 
    
    series = df[col].dropna()
    if not series.empty:
        ymin = series.min()
        ymax = series.max()
        pad = (ymax - ymin) * 0.05 if ymax != ymin else 1.0
        ax.set_ylim(bottom=0)

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Tufte-style: remove top/right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Optional: Add a single legend to the first plot if needed
# axes[0].legend(frameon=False)

plt.show()
fig.savefig("drl_caching_metrics.png", dpi=300)


# In[ ]:


if __name__ == "__main__":
    cfg = config.Config()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True
    })

    COLORS = ["#2b7bba", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    folder = r"c:\Users\es25591\Workspace\CacheVideoPredict360\Results"
    folder= cfg.path_results
    
    files = [
        "drl_pan_eps0.985_lrdecay0.9995_c50_ar200.0_z0.8.csv",
        "drl_pan_eps0.98_lrdecay0.9999_c50_ar200.0_z0.8.csv",
        "drl_pan_eps0.97_lrdecay0.9997_c50_ar200.0_z0.8.csv",
        # "drl_pan_eps0.97_lrdecay0.9997_c100_ar200.0_z0.8.csv"
    ]

    paths = [os.path.join(folder, f) for f in files]
    def _short_label(fname: str) -> str:
        stem = os.path.splitext(fname)[0]
        if "lrdecay" in stem:
            val = stem.split("lrdecay", 1)[1].split("_", 1)[0]
            return f"lr decay: {val}"
        if "fixedlr" in stem:
            val = stem.split("fixedlr", 1)[1].split("_", 1)[0]
            return f"fixed lr: {val}"
        return stem

    labels = [_short_label(f) for f in files]

    print(labels)
    metrics = ["total_reward", "hit_rate", "miss_rate", "epsilon", "lr"]
    window_size = 1

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3), sharex=True)

    for color, path, label in zip(COLORS, paths, labels):
        df = pd.read_csv(path).head(100).copy()   # only first 100 values
        total = (df["cache_hits"] + df["cache_misses"]).replace(0, np.nan)
        df["hit_rate"] = (df["cache_hits"] / total) * 100
        df["miss_rate"] = (df["cache_misses"] / total) * 100

        for ax, col in zip(axes, metrics):
            ax.plot(df["episode"], df[col], color=color, alpha=0.25, linewidth=0.8)

            if col != "epsilon":
                smoothed = df[col].rolling(window=window_size).mean()
                ax.plot(df["episode"], smoothed, color=color, linewidth=1.5, label=label)
            else:
                ax.plot(df["episode"], df[col], color=color, linewidth=1.5, label=label)

    for ax, col in zip(axes, metrics):
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylim(bottom=0)

    axes[0].legend(frameon=False, loc="best")
    plt.show()

    fig.savefig("interval_04_comparison.png", dpi=300)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------      
# 1. Publication Style Configuration
# ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True
})

COLORS = ["#2b7bba", "#d62728"]

# ----------------------------------------------------------------
# 2. Data Preparation with Variance
# ----------------------------------------------------------------
def get_metrics_with_error(path):
    """Returns mean and standard deviation for hit and miss rates."""
    try:
        df = pd.read_csv(path)
        total = (df["cache_hits"] + df["cache_misses"]).replace(0, np.nan)
        
        hit_series = (df["cache_hits"] / total * 100)
        miss_series = (df["cache_misses"] / total * 100)
        
        # Calculate Mean and Standard Deviation
        return (hit_series.mean(), hit_series.std()), (miss_series.mean(), miss_series.std())
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return (0, 0), (0, 0)

drl_files = {
    "5": r"c:\Users\es25591\Workspace\CacheVideoPredict360\Results\drl_pan_opt_c25_ar10.0_z0.5.csv",
    "10": r"c:\Users\es25591\Workspace\CacheVideoPredict360\Results\drl_pan_opt_c50_ar10.0_z0.5.csv",
}

lsr_files = {
    "5": r"c:\Users\es25591\Workspace\CacheVideoPredict360\Results\lru_c25_ar10.0_z0.5.csv",
    "10": r"c:\Users\es25591\Workspace\CacheVideoPredict360\Results\lru_c50_ar10.0_z0.5.csv",
}

caps = list(drl_files.keys())
# Structure: { Metric: { Algo: [means], Algo_err: [stds] } }
stats = {
    "Hit Rate": {"DRL-LSR": [], "DRL-LSR_err": [], "LRU-LSR": [], "LRU-LSR_err": []},
    "Miss Rate": {"DRL-LSR": [], "DRL-LSR_err": [], "LRU-LSR": [], "LRU-LSR_err": []}
}

for c in caps:
    (d_h, d_h_err), (d_m, d_m_err) = get_metrics_with_error(drl_files[c])
    (l_h, l_h_err), (l_m, l_m_err) = get_metrics_with_error(lsr_files[c])
    
    stats["Hit Rate"]["DRL-LSR"].append(d_h)
    stats["Hit Rate"]["DRL-LSR_err"].append(d_h_err)
    stats["Hit Rate"]["LRU-LSR"].append(l_h)
    stats["Hit Rate"]["LRU-LSR_err"].append(l_h_err)
    
    stats["Miss Rate"]["DRL-LSR"].append(d_m)
    stats["Miss Rate"]["DRL-LSR_err"].append(d_m_err)
    stats["Miss Rate"]["LRU-LSR"].append(l_m)
    stats["Miss Rate"]["LRU-LSR_err"].append(l_m_err)

# ----------------------------------------------------------------
# 3. Plotting logic
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
x = np.arange(len(caps))
width = 0.35 

# Error bar styling
error_kw = dict(lw=1, capsize=3, capthick=1, ecolor='#333333')

metrics = ["Hit Rate", "Miss Rate"]
for i, metric in enumerate(metrics):
    ax = axes[i]
    
    ax.bar(x - width/2, stats[metric]["DRL-LSR"], width, 
           yerr=stats[metric]["DRL-LSR_err"], error_kw=error_kw,
           label="DRL-LSR", color=COLORS[0], edgecolor='black', linewidth=0.6, zorder=3)
    
    ax.bar(x + width/2, stats[metric]["LRU-LSR"], width, 
           yerr=stats[metric]["LRU-LSR_err"], error_kw=error_kw,
           label="LRU-LSR", color=COLORS[1], edgecolor='black', linewidth=0.6, zorder=3)

    ax.set_title(f"Average {metric}", fontweight="bold")
    ax.set_xlabel("Cache Size (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(caps)
    ax.set_ylim(0, 110) # Increased to accommodate error bars
    
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

axes[0].set_ylabel("Percentage (%)")
axes[1].legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.show()

fig.savefig("drl_vs_lru_cache_performance.pdf")

