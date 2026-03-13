#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from time import sleep
from collections import deque, defaultdict
from itertools import count
from typing import Any, Dict, Counter, List

sources_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if sources_path not in sys.path:
    sys.path.append(sources_path)

from importnb import Notebook
with Notebook():
    from Labs.LatencyModel import LatencyModel, MultiDULatencyModel
    from Labs.Policy import DrlPolicy, MMSPPolicy
    from Labs.CacheEngine import CacheEngineEnv
    from Labs.UserRequest import UserRequestEvents
    from Labs.EnvWrapper import EnvWrapper

from RL.Networks import QNetwork, MultiHeadQNetwork
from RL.Buffers import ReplayBuffer, NStepReplayBuffer
from RL.Adapters import FeatureAdapter, NetworkAdapter
from RL.FocusWorkers import BaseWorker, EnhWorker, FocusWorker, PDQNWorker

import Common.config as config
import Common.datatypes as datatypes
import Common.debugger as debugger
import Common.utils as utils
import Core.builders as builders


importlib.reload(builders)
importlib.reload(config)
importlib.reload(datatypes)
importlib.reload(debugger)
importlib.reload(utils)


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UserTransition = datatypes.UserTransition
CachePolicy = datatypes.CachePolicy
CacheKey = datatypes.CacheKey

cfg = config.Config()
cfg.filename = \
    f"focus_eps{cfg.epsilon_start}_" \
    f"lrdecay{cfg.learning_rate_decay}_" \
    f"gamma{cfg.gamma}.csv"
cfg.state_dim_base_focus = cfg.state_dim_base_focus + cfg.state_dim_enh_focus
cfg.state_dim = 10 * cfg.cache_size + 2 * (cfg.viewport + 1)
cfg.action_dim = cfg.cache_size * (cfg.viewport + 1) + 1
cfg.hidden_dim = 256

debugger = debugger.debug


# In[3]:


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

def update_metrics(info: dict, reward: float) -> tuple[float, int, int, int, int]:
    enh_hits = info.get("enh_layer_hits", 0)
    base_hits = info.get("base_layer_hits", 0)
    enh_misses = info.get("enh_layer_misses", 0)
    base_misses = info.get("base_layer_misses", 0)

    return reward, base_hits, base_misses, enh_hits, enh_misses


# In[4]:


def run_episode(episode, env, agent, net_adapter, cfg):
    """Run one full training episode."""
    _, info = net_adapter.reset()

    total_reward = 0.0
    cache_hits = cache_misses = 0
    base_hits = base_misses = 0
    enh_hits = enh_misses = 0

    if cfg.has_warmup:
        env.warmup_phase(net_adapter, 1000)

    for step in range(cfg.max_steps):
        
        req_state = info.get("user_request", None)
        
        # --- Build State ---
        state = net_adapter.build_observation(req_state)

        # --- Action Selection --- 
        action = agent.select_action(state)
        
        # --- Environment Step ---
        _, _, done, info = env.step(action, req_state, net_adapter)

        # --- Store Transition & Train ---
        nxt_req = info["user_request"]

        reward_0 = info["reward_layer_0"]
        reward_1 = info["reward_layer_1"]
        
        prefetch_base = info["prefetch_base"]
        prefetch_enh = info["prefetch_enh"]

        next_state = net_adapter.build_observation(nxt_req)

        if prefetch_base or prefetch_enh:
            agent.remember(
                state, action, reward_0 + reward_1, next_state, done
            )
            agent.train_step()

        delta_r, bs_hits, bs_miss, e_hits, e_miss = update_metrics(info, reward_0 + reward_1)
        total_reward += delta_r
        cache_hits += bs_hits + e_hits
        cache_misses += bs_miss + e_miss
        base_hits += bs_hits
        base_misses += bs_miss
        enh_hits += e_hits
        enh_misses += e_miss

        if done:
            break

        debugger.log('cache_hits', bs_hits + e_hits)
        debugger.log('cache_misses', bs_miss + e_miss)

    return total_reward, cache_hits, cache_misses, base_hits, base_misses, enh_hits, enh_misses

def train(cfg):
    env = builders.build_environment(cfg)

    # agent = FocusWorker(cfg, debugger=debugger)

    agent = PDQNWorker(cfg, debugger=debugger)

    feature_adapter = FeatureAdapter(cfg, env)
    net_adapter = NetworkAdapter(cfg, env, feature_adapter)

    date_dir = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    debug_path = os.path.join(cfg.path_results, date_dir)
    os.makedirs(debug_path, exist_ok=True)

    print(f"Starting training for {cfg.n_episodes} episodes... {date_dir}")
    print(f"Warmup Phase: {'Enabled' if cfg.has_warmup else 'Disabled'}")
    print(f"Users Session Length: {cfg.user_session_length}")
    print(
        f"State Dim Base: {agent.state_dim}, Action Dim Base: {agent.action_dim_base}, Hidden Dim Base: {cfg.hidden_dim_base_focus}\n"
        f"State Dim Enh: {agent.state_dim}, Action Dim Enh: {agent.action_dim_enh}, Hidden Dim Enh: {cfg.hidden_dim_enh_focus}"
    )

    for episode in range(cfg.n_episodes):

        total_reward, hits, misses, bs_hits, bs_miss, enh_hits, enh_miss = run_episode(
            episode, env, agent, net_adapter, cfg
        )

        agent.update_epsilon()

        save_training_results(
            path_=cfg.path_results + "/" + date_dir,
            filename=cfg.filename,
            ep=episode,
            total_reward=total_reward,
            cache_hits=hits,
            cache_misses=misses,
            agent=agent
        )

        debugger.log('lr', agent.scheduler.get_last_lr()[0])
        debugger.log('epsilon', agent.epsilon)

        print(
            f"--- Episode {episode} | R: {total_reward:.2f} | "
            f"HR: {hits / (hits + misses + 1e-9):.2f} | "
            f"BHR: {bs_hits / (bs_hits + bs_miss + 1e-9):.2f} | "
            f"EHR: {enh_hits / (enh_hits + enh_miss + 1e-9):.2f} ---"
        )

        debugger.save_results(filepath=f"{debug_path}/debug_ep{episode}")
        debugger.clear()

        print("-" * 50)

    return agent
    
if __name__ == "__main__":
    agent = train(cfg)


# In[ ]:


if __name__ == "__main__":

    cfg_optimal_100 = cfg
    cfg_optimal_100.n_episodes = 100

    # Force greedy action selection (no exploration)
    cfg_optimal_100.epsilon_start = 0.00
    cfg_optimal_100.epsilon_min = 0.00
    cfg_optimal_100.epsilon_decay = 1.0

    cfg_optimal_100.filename = (
        f"pdqn_optimal_eps0.0_lrdecay{cfg_optimal_100.learning_rate_decay}_"
        f"gamma{cfg_optimal_100.gamma}_100ep.csv"
    )
    cfg = cfg_optimal_100

    env = builders.build_environment(cfg)

    feature_adapter = FeatureAdapter(cfg, env)
    net_adapter = NetworkAdapter(cfg, env, feature_adapter)

    date_dir = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    debug_path = os.path.join(cfg.path_results, date_dir)
    os.makedirs(debug_path, exist_ok=True)

    print(f"Starting training for {cfg.n_episodes} episodes... {date_dir}")
    print(f"Warmup Phase: {'Enabled' if cfg.has_warmup else 'Disabled'}")
    print(f"Users Session Length: {cfg.user_session_length}")
    print(
        f"State Dim Base: {agent.state_dim}, Action Dim Base: {agent.action_dim_base}, Hidden Dim Base: {cfg.hidden_dim_base_focus}\n"
        f"State Dim Enh: {agent.state_dim}, Action Dim Enh: {agent.action_dim_enh}, Hidden Dim Enh: {cfg.hidden_dim_enh_focus}"
    )

    for episode in range(cfg.n_episodes):

        total_reward, hits, misses, bs_hits, bs_miss, enh_hits, enh_miss = run_episode(
            episode, env, agent, net_adapter, cfg
        )

        agent.update_epsilon()

        save_training_results(
            path_=cfg.path_results + "/" + date_dir,
            filename=cfg.filename,
            ep=episode,
            total_reward=total_reward,
            cache_hits=hits,
            cache_misses=misses,
            agent=agent
        )

        debugger.log('lr', agent.scheduler.get_last_lr()[0])
        debugger.log('epsilon', agent.epsilon)

        print(
            f"--- Episode {episode} | R: {total_reward:.2f} | "
            f"HR: {hits / (hits + misses + 1e-9):.2f} | "
            f"BHR: {bs_hits / (bs_hits + bs_miss + 1e-9):.2f} | "
            f"EHR: {enh_hits / (enh_hits + enh_miss + 1e-9):.2f} ---"
        )

        debugger.save_results(filepath=f"{debug_path}/debug_ep{episode}")
        debugger.clear()

        print("-" * 50)

