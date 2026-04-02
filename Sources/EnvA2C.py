#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
    from Labs.Policy import DrlPolicy
    from Labs.CacheEngine import CacheEngineEnv
    from Labs.UserRequest import UserRequestEvents
    from Labs.EnvWrapper import EnvWrapper

from RL.Networks import QNetwork, MultiHeadQNetwork
from RL.Buffers import ReplayBuffer, NStepReplayBuffer
from RL.Adapters import FeatureAdapter, NetworkAdapter
from RL.FocusWorkers import BaseWorker, EnhWorker, FocusWorker
from RL.A2CWorker import A2CWorker

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


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UserTransition = datatypes.UserTransition
CachePolicy = datatypes.CachePolicy
CacheKey = datatypes.CacheKey

# Note: The state and action dimensions are determined by the cache size and the specific 
# design of the state representation and action space. Adjust these calculations based on 
# your actual implementation of the state and action spaces.
cfg = config.Config()
cfg.filename = \
    f"focus_eps{cfg.epsilon_start}_" \
    f"lrdecay{cfg.learning_rate_decay}_" \
    f"gamma{cfg.gamma}.csv"
cfg.state_dim_base_focus = cfg.cache_size * 10 + 2
cfg.state_dim_enh_focus = cfg.cache_size * 10 + 2
cfg.action_dim_base_focus = cfg.cache_size * 5 + 1

debugger = debugger.debug


# In[ ]:


class NetworkAdapter:
    def __init__(self, cfg: Any, env: Any, feature_adapter: Any):
        self.env = env
        self.cfg = cfg
        self.features = feature_adapter

        self.C = self.cfg.cache_size  # paper's cache capacity (videos)
        self.k = self.cfg.viewport    # paper's tiles per video (enhancement)

    def build_observation(self, idx_vp, video, tile = None) -> np.ndarray:

        cache = self.env.mec_cache.policy.cache

        x_s = np.zeros(len(cache), dtype=np.float32)
        x_l = np.zeros(len(cache), dtype=np.float32)

        for idx, (v, t) in enumerate(cache):
            if v == -1:
                continue

            if t == -1:
                x_s[idx] = self.features.video_freq_short.get(v, 0) / self.features.video_hist_short.maxlen
                x_l[idx] = self.features.video_freq_long.get(v, 0) / self.features.video_hist_long.maxlen
            else:
                x_s[idx] = self.features.tile_freq_short.get((v, t), 0) / self.features.tile_hist_short.maxlen
                x_l[idx] = self.features.tile_freq_long.get((v, t), 0) / self.features.tile_hist_long.maxlen

        if tile is None:
            y_s = np.array(
                [self.features.video_freq_short.get(video, 0) / self.features.video_hist_short.maxlen], 
                dtype=np.float32
            )
            y_l = np.array(
                [self.features.video_freq_long.get(video, 0) / self.features.video_hist_long.maxlen], 
                dtype=np.float32
            )
        else:
            y_s = np.array(
                [self.features.tile_freq_short.get((video, tile), 0) / self.features.tile_hist_short.maxlen], 
                dtype=np.float32
            )
            y_l = np.array(
                [self.features.tile_freq_long.get((video, tile), 0) / self.features.tile_hist_long.maxlen], 
                dtype=np.float32
            )

        step_one_hot = int(tile is None)
        # arr_idx = np.zeros(5, dtype=np.int32)
        # arr_idx[idx_vp] = 1

        return np.concatenate(
            [x_s, x_l, y_s, y_l], 
            axis=0
        )

    def reset(self):
        
        obs, info = self.env.reset()
        self.features.reset_history()

        return obs, info
    
    def env_is_done(self) -> bool:
        return self.env.users_env.all_users_done()


# In[9]:


def _append_csv_row(csv_path: str, fieldnames: list[str], row: dict) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_episode_metrics(
    metrics_dir: str,
    ep: int,
    total_reward: float,
    cache_hits: int,
    cache_misses: int,
    agent,
):
    csv_path = os.path.join(metrics_dir, 'episode_metrics.csv')
    fieldnames = [
        'episode',
        'total_reward',
        'cache_hits',
        'cache_misses',
        'hit_rate',
        'epsilon',
        'lr',
    ]

    row = {
        'episode': ep,
        'total_reward': round(float(total_reward), 2),
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'hit_rate': float(cache_hits) / float(cache_hits + cache_misses + 1e-9),
        'epsilon': round(float(agent.epsilon), 6) if agent else None,
        'lr': float(agent.scheduler.get_last_lr()[0]) if agent else None,
    }
    _append_csv_row(csv_path, fieldnames, row)

def save_step_metrics(
    metrics_dir: str,
    episode: int,
    episode_step: int,
    global_step: int,
    reward: float,
    agent,
    train_metrics: dict | None,
):
    csv_path = os.path.join(metrics_dir, 'step_metrics.csv')
    fieldnames = [
        'episode',
        'episode_step',
        'global_step',
        'reward',
        'epsilon',
        'lr',
        'train_loss',
        'actor_loss',
        'critic_loss'
    ]

    row = {
        'episode': episode,
        'episode_step': episode_step,
        'global_step': global_step,
        'reward': float(reward),
        'epsilon': round(float(agent.epsilon), 6) if agent else None,
        'lr': float(agent.scheduler.get_last_lr()[0]) if agent else None,
        'train_loss': None,
        'actor_loss': None,
        'critic_loss': None
    }

    if train_metrics is not None:
        row.update({
            'train_loss': train_metrics.get('train_loss'),
            'actor_loss': train_metrics.get('actor_loss'),
            'critic_loss': train_metrics.get('critic_loss'),
    })

    _append_csv_row(csv_path, fieldnames, row)

def update_metrics(info: dict, reward: float) -> tuple[float, int, int, int, int]:
    enh_hits = info.get("enh_layer_hits", 0)
    base_hits = info.get("base_layer_hits", 0)
    enh_misses = info.get("enh_layer_misses", 0)
    base_misses = info.get("base_layer_misses", 0)

    return reward, base_hits, base_misses, enh_hits, enh_misses


# In[ ]:


def select_action(agent, req_state, env, net_adapter=None):

    if req_state is None:
        return None, np.zeros(5, dtype=np.int32), [None] * (1 + cfg.viewport)

    backhaul_usage = 0
    missing = env._missing_items(req_state)
    transition = [None] * (1 + cfg.viewport)

    if missing[0] == 1:
        state_base = net_adapter.build_observation(0, req_state["video"])
        action_base, value_base, prob_base = agent.select_action(state_base)

        transition[0] = {
            'state': state_base,
            'action': action_base,
            'value': value_base,
            'prob': prob_base
        }

        message = {
            "video": req_state["video"],
            "tiles": [],
            "base_req_init": True,
            "action_idx": action_base,
        }
        env.prefetch_fn(env.mec_cache, message)

        if action_base != 0:
            backhaul_usage += 12 * env.mec_cache.tile_size_bytes[0]

        debugger.log("base_action", action_base)

    for idx, missing_item in enumerate(missing[1:]):
        if missing_item == 1:
            state_enh = net_adapter.build_observation(
                idx + 1,
                req_state["video"],
                req_state["viewport"][idx]
            )
            action_enh, value_enh, prob_enh = agent.select_action(state_enh)

            transition[idx + 1] = {
                'state': state_enh,
                'action': action_enh,
                'value': value_enh,
                'prob': prob_enh
            }

            message = {
                "video": req_state["video"],
                "tiles": [req_state["viewport"][idx]],
                "base_req_init": False,
                "action_idx": action_enh,
            }
            env.prefetch_fn(env.mec_cache, message)

            if action_enh != 0:
                backhaul_usage += env.mec_cache.tile_size_bytes[1]

            debugger.log(f"enh_{idx}_action", action_enh)

    debugger.log("backhaul_usage", backhaul_usage)

    debugger.log("base_layer_miss", missing[0])
    for i, is_missing in enumerate(missing[1:], start=1):
        debugger.log(f"enh_layer_missing_{i}", is_missing)

    return None, missing, transition


def run_episode(episode, env, agent, net_adapter, cfg, metrics_dir, global_step_start):
    """Run one full training episode and persist step-level metrics."""
    _, info = net_adapter.reset()

    total_reward = 0.0
    cache_hits = cache_misses = 0
    base_hits = base_misses = 0
    enh_hits = enh_misses = 0
    psnr_sum = 0.0

    global_step = global_step_start

    if cfg.has_warmup:
        env.warmup_phase(net_adapter, 1000)

    if episode == 0:
        max_steps = 10000
    else:
        max_steps = 10000

    for step in range(max_steps):
        global_step += 1

        # --- Build State ---
        req_state = info.get("user_request", None)

        # --- Action Selection ---
        action, missing, transition = select_action(agent, req_state, env, net_adapter)

        # --- Environment Step ---
        _, reward, done, info = env.step(action, req_state, net_adapter)

        # --- Store Transition & Train ---
        nxt_req = info["user_request"]

        weight = 0.7
        reward_0 = info["reward_layer_0"]
        reward_1 = info["reward_layer_1"]

        # reward_0 = reward_0/30.0
        # reward_1 = reward_1/10.0

        # reward = weight * reward_0 + (1 - weight) * reward_1
        reward = ((info["psnr"] - 30.0)/10.0)

        if reward < 0:
            print(f"Negative reward at step {step}: {reward:.2f} | PSNR: {info['psnr']:.2f}")
            print(f"Step {step} | U: {nxt_req['u']} | V: {nxt_req['video']} | GOP: {nxt_req['gop']} | RL0: {reward_0:.2f} | RL1: {reward_1:.2f} | BH: {info['base_layer_hits']} | EH: {info['enh_layer_hits']} | BM: {info['base_layer_misses']} | EM: {info['enh_layer_misses']}")
            
        queued_update = False

        if missing[0] == 1 or transition[0] is not None:
            next_state_base = net_adapter.build_observation(0, nxt_req["video"])

            agent.remember(
                transition[0]['state'],
                transition[0]['action'],
                reward,
                next_state_base,
                done
            )
            queued_update = True

        for i in range(len(missing) - 1):
            if missing[i + 1] == 1 and transition[i + 1] is not None:
                next_state_enh = net_adapter.build_observation(i + 1, nxt_req["video"], nxt_req["viewport"][i])
                agent.remember(
                    transition[i + 1]['state'],
                    transition[i + 1]['action'],
                    reward,
                    next_state_enh,
                    done
                )
                queued_update = True

        if queued_update:# and episode == 0:
            train_metrics = agent.train_step()
        
            save_step_metrics(
                metrics_dir=metrics_dir,
                episode=episode,
                episode_step=step,
                global_step=global_step,
                reward=reward,
                agent=agent,
                train_metrics=train_metrics,
            )

        delta_r, bs_hits, bs_miss, e_hits, e_miss = update_metrics(info, reward)
        total_reward += delta_r
        cache_hits += bs_hits + e_hits
        cache_misses += bs_miss + e_miss
        base_hits += bs_hits
        base_misses += bs_miss
        enh_hits += e_hits
        enh_misses += e_miss
        psnr_sum += info.get("psnr", 0.0)

        if done:
            break

        debugger.log('cache_hits', bs_hits + e_hits)
        debugger.log('cache_misses', bs_miss + e_miss)

    return total_reward, cache_hits, cache_misses, base_hits, base_misses, enh_hits, enh_misses, global_step, psnr_sum / (step + 1)

def train(cfg):
    env = builders.build_environment(cfg)

    agent = A2CWorker(cfg, debugger=debugger)

    feature_adapter = FeatureAdapter(cfg, env)
    net_adapter = NetworkAdapter(cfg, env, feature_adapter)

    date_dir = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    debug_path = os.path.join(cfg.path_results, date_dir)
    metrics_dir = os.path.join(debug_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Starting training for {cfg.n_episodes} episodes... {date_dir}")
    print(f"Warmup Phase: {'Enabled' if cfg.has_warmup else 'Disabled'}")
    print(f"Users Session Length: {cfg.user_session_length}")
    print(
        f"State Dim Base: {agent.state_dim}, Action Dim Base: {agent.action_dim}, Hidden Dim Base: {cfg.hidden_dim_base_focus}"
    )

    global_step = 0

    for episode in range(cfg.n_episodes):

        total_reward, hits, misses, bs_hits, bs_miss, enh_hits, enh_miss, global_step, psnr_rate = run_episode(
            episode, env, agent, net_adapter, cfg, metrics_dir, global_step
        )

        agent.update_epsilon() # Function used to clean buffer memory

        save_episode_metrics(
            metrics_dir=metrics_dir,
            ep=episode,
            total_reward=total_reward,
            cache_hits=hits,
            cache_misses=misses,
            agent=agent,
        )

        debugger.log('lr', agent.scheduler.get_last_lr()[0])
        debugger.log('epsilon', agent.epsilon)

        debugger.save_results(filepath=f"{debug_path}/debug_ep{episode}")
        debugger.clear()

        print(
            f"Episode {episode} | R: {int(total_reward)} | "
            f"HR: {hits / (hits + misses + 1e-9):.2f} | "
            f"BHR: {bs_hits / (bs_hits + bs_miss + 1e-9):.2f} | "
            f"EHR: {enh_hits / (enh_hits + enh_miss + 1e-9):.2f} | "
            f"PSNR: {psnr_rate:.2f} | "
            f"Time: {pd.Timestamp.now().strftime('%H:%M:%S')}"
        )
        print("-" * 50)

if __name__ == "__main__":
    train(cfg)

