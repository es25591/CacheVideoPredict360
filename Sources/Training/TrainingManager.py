
import os
import sys
import numpy as np

from itertools import count
from typing import Any, Dict, Tuple

sources_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if sources_path not in sys.path:
    sys.path.append(sources_path)

from RL.EnhWorker import EnhWorker
from RL.BaseWorker import BaseWorker
from RL.Adapters import FeatureAdapter, NetworkAdapter

import Common.debugger as debugger
import Core.builders as builders


class TrainingManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = builders.build_environment(self.cfg)
        self.base_agent = BaseWorker(self.cfg, debugger=debugger.debug)
        self.enh_agent = EnhWorker(self.cfg, debugger=debugger.debug)

        self.feature_adapter = FeatureAdapter(self.cfg, self.env)
        self.net_adapter = NetworkAdapter(
            self.cfg, self.env, self.feature_adapter
        )

        self.ep_rewards = []
        self.ep_cache_hits = []
        self.ep_cache_misses = []

        self.k_base = 600.0  # Scales probability delta up to PSNR 30 bound
        self.k_enh = 250.0  # Scales probability delta up to PSNR 10 bound

    def select_action(self, state):
        state_base, state_enh = state
        action_base = self.base_agent.select_action(state_base)
        action_enh = self.enh_agent.select_action(state_enh)

        return np.concatenate([[action_base], action_enh], axis=0) 
    
    def train_fn(self):

        total_reward = 0.0
        cache_hits = cache_misses = 0
        base_hits = base_misses = 0
        enh_hits = enh_misses = 0

        _, info = self.net_adapter.reset()

        self.env.warmup_phase(self.net_adapter)

        for step in count():
            req = info["user_request"]
            state = self.net_adapter.build_observation(req)

            # --- Action Selection --- 
            action = self.select_action(state)
                        
            # --- Environment Step ---
            _, reward, done, info = self.env.step(
                action, req, self.net_adapter
            )

            nxt_req = info["user_request"]

            reward_0 = info["reward_layer_0"]
            reward_1 = info["reward_layer_1"]
            prefetch_base = info["prefetch_base"]
            prefetch_enh = info["prefetch_enh"]
                    
            state_base, state_enh = state
            next_state_base, next_state_enh = self.net_adapter.build_observation(nxt_req)
    
            if prefetch_base:
                self.base_agent.remember(
                    state_base, action[0], reward_0, next_state_base, done
                )
                self.base_agent.train_step()

            if prefetch_enh:
                self.enh_agent.remember(
                    state_enh, action[1:], reward_1, next_state_enh, done
                )
                self.enh_agent.train_step()

            # --- Update Metrics ---
            delta_r, bs_hits, bs_miss, e_hits, e_miss = self.update_metrics(info, reward)
            total_reward += delta_r
            cache_hits += bs_hits + e_hits
            cache_misses += bs_miss + e_miss
            base_hits += bs_hits
            base_misses += bs_miss
            enh_hits += e_hits
            enh_misses += e_miss

            if done:
                break
            
        return total_reward, cache_hits, cache_misses, base_hits, base_misses, enh_hits, enh_misses

    def update_metrics(self, info: Dict[str, Any], reward: float) -> Tuple[float, int, int, int, int]:
        """
        Extract cache hit/miss metrics from env info dict.

        Returns:
            (reward, base_hits, base_misses, enh_hits, enh_misses)
        """
        enh_hits = int(info.get("enh_layer_hits", 0))
        base_hits = int(info.get("base_layer_hits", 0))
        enh_misses = int(info.get("enh_layer_misses", 0))
        base_misses = int(info.get("base_layer_misses", 0))
        
        return reward, base_hits, base_misses, enh_hits, enh_misses
