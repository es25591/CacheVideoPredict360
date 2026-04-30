import csv
import importlib
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from typing import Any

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
from RL.Adapters import FeatureAdapter
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


# The state and action dimensions are derived from the cache layout and the
# observation design used by the agent.
cfg = config.Config()
cfg.filename = \
    f"focus_eps{cfg.epsilon_start}_" \
    f"lrdecay{cfg.learning_rate_decay}_" \
    f"gamma{cfg.gamma}.csv"
cfg.state_dim = cfg.cache_size * 10 + 4
cfg.action_dim = cfg.cache_size * 5 + 1

debugger = debugger.debug

CONTENT_MISS_ID = -1
OBSERVATION_TAIL_DIM = 5


def _agent_epsilon(agent: Any) -> float | None:
    return round(float(agent.epsilon), 6) if agent else None


def _agent_learning_rate(agent: Any) -> float | None:
    return float(agent.scheduler.get_last_lr()[0]) if agent else None


def _safe_hit_rate(hits: int, misses: int) -> float:
    return float(hits) / float(hits + misses + 1e-9)


def _normalized_count(value: int, history) -> float:
    return float(value) / float(history.maxlen)


class NetworkAdapter:
    """Build flat observations from cache state and the current request."""

    def __init__(self, cfg: Any, env: Any, feature_adapter: Any):
        self.env = env
        self.cfg = cfg
        self.features = feature_adapter

        self.cache_capacity = self.cfg.cache_size
        self.viewport_size = self.cfg.viewport
        
        self.features.reset_history()

    def _encode_cache_slot(self, video: int, tile: int) -> tuple[float, float]:
        if video == CONTENT_MISS_ID:
            return 0.0, 0.0

        if tile == CONTENT_MISS_ID:
            return (
                _normalized_count(
                    self.features.video_freq_short.get(video, 0),
                    self.features.video_hist_short,
                ),
                _normalized_count(
                    self.features.video_freq_long.get(video, 0),
                    self.features.video_hist_long,
                ),
            )

        return (
            _normalized_count(
                self.features.tile_freq_short.get((video, tile), 0),
                self.features.tile_hist_short,
            ),
            _normalized_count(
                self.features.tile_freq_long.get((video, tile), 0),
                self.features.tile_hist_long,
            ),
        )

    def _encode_request_tail(self, video: int, tile: int | None) -> tuple[np.ndarray, np.ndarray]:
        if tile is None:
            return (
                np.array([
                    _normalized_count(
                        self.features.video_freq_short.get(video, 0),
                        self.features.video_hist_short,
                    )
                ], dtype=np.float32),
                np.array([
                    _normalized_count(
                        self.features.video_freq_long.get(video, 0),
                        self.features.video_hist_long,
                    )
                ], dtype=np.float32),
            )

        return (
            np.array([
                _normalized_count(
                    self.features.tile_freq_short.get((video, tile), 0),
                    self.features.tile_hist_short,
                )
            ], dtype=np.float32),
            np.array([
                _normalized_count(
                    self.features.tile_freq_long.get((video, tile), 0),
                    self.features.tile_hist_long,
                )
            ], dtype=np.float32),
        )

    def build_observation(self, tile_idx, video, tile=None) -> np.ndarray:
        """Return the flat observation vector consumed by the agent."""

        cache = self.env.mec_cache.policy.cache

        short_hist = np.zeros(len(cache), dtype=np.float32)
        long_hist = np.zeros(len(cache), dtype=np.float32)

        for idx, (v, t) in enumerate(cache):
            if v != CONTENT_MISS_ID:
                short_value, long_value = self._encode_cache_slot(v, t)
                short_hist[idx] = short_value
                long_hist[idx] = long_value

        request_short, request_long = self._encode_request_tail(video, tile)

        request_kind = np.zeros(OBSERVATION_TAIL_DIM, dtype=np.float32)
        if tile_idx is None:
            request_kind[0] = 1.0
        else:
            request_kind[tile_idx + 1] = 1.0

        return np.concatenate([short_hist, long_hist, request_short, request_long, request_kind], axis=0)

    def reset(self):
        obs, info = self.env.reset()
        return obs, info
    
    def env_is_done(self) -> bool:
        return self.env.users_env.all_users_done()


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
    fieldnames = ['episode', 'episode_step', 'global_step', 'reward', 'epsilon', 'lr', 'train_loss', 'actor_loss', 'critic_loss']

    row = {
        'episode': episode,
        'episode_step': episode_step,
        'global_step': global_step,
        'reward': float(reward),
        'epsilon': _agent_epsilon(agent),
        'lr': _agent_learning_rate(agent),
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


def save_episode_models(debug_path: str, episode: int, agent) -> str | None:
    models_dir = os.path.join(debug_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    network_params = {"episode": episode, "networks": {}}

    for net_name in ("q_network", "policy_net", "actor", "critic", "model", "network"):
        net = getattr(agent, net_name, None)
        if net is not None and hasattr(net, "state_dict"):
            network_params["networks"][net_name] = {
                k: v.detach().cpu().tolist() for k, v in net.state_dict().items()
            }

    if not network_params["networks"]:
        return None

    json_path = os.path.join(models_dir, f"network_params_ep{episode}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(network_params, f, indent=2)

    return json_path


def _content_key(video: int, tile: int | None) -> tuple[int, int]:
    return (int(video), CONTENT_MISS_ID if tile is None else int(tile))

def _content_label(content_key: tuple[int, int]) -> str:
    video, tile = content_key
    if tile == -1:
        return f"V{video}-Base"
    return f"V{video}-T{tile}"

def _snapshot_cache(env) -> set[tuple[int, int]]:
    cache_set: set[tuple[int, int]] = set()
    cache_entries = getattr(env.mec_cache.policy, "cache", [])

    for video, tile in cache_entries:
        if int(video) == CONTENT_MISS_ID:
            continue
        cache_set.add((int(video), int(tile)))

    return cache_set

def plot_topk_requested_with_cache(
    request_counts: dict[tuple[int, int], int],
    cache_snapshot: set[tuple[int, int]],
    episode: int,
    top_k: int,
    out_dir: str,
) -> None:
    if not request_counts:
        return

    sorted_items = sorted(
        request_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:max(1, int(top_k))]

    labels = [_content_label(key) for key, _ in sorted_items]
    values = [count for _, count in sorted_items]
    colors = ["#2ca02c" if key in cache_snapshot else "#1f77b4" for key, _ in sorted_items]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    bars = ax.bar(labels, values, color=colors)

    ax.set_title(f"Episode {episode}: Top-{len(sorted_items)} Requested Content")
    ax.set_xlabel("Content")
    ax.set_ylabel("Request count")
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#2ca02c", label="In cache (episode end)"),
            Patch(facecolor="#1f77b4", label="Not in cache"),
        ]
    )

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"topk_requested_ep{episode}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def log_selection_debug(transition, missing, backhaul_usage):
    if transition[0] is not None:
        debugger.log("base_action", transition[0]["action"])
        debugger.log("base_value", transition[0]["value"])
        debugger.log("base_prob", transition[0]["prob"])
        debugger.log("base_entropy", transition[0]["entropy"])

    for idx, t in enumerate(transition[1:]):
        if t is not None:
            debugger.log(f"enh_{idx}_action", t["action"])
            debugger.log(f"enh_{idx}_value", t["value"])
            debugger.log(f"enh_{idx}_prob", t["prob"])
            debugger.log(f"enh_{idx}_entropy", t["entropy"])

    debugger.log("backhaul_usage", backhaul_usage)

    debugger.log("base_layer_miss", missing[0])
    for i, is_missing in enumerate(missing[1:], start=1):
        debugger.log(f"enh_layer_missing_{i}", is_missing)

def process_missing_layer(
    agent,
    req_state,
    env,
    net_adapter,
    missing,
    transition,
    backhaul_usage: int,
    layer_idx: int,
    tile_idx: int | None,
    is_base: bool,
    bytes_cost: int,
) -> int:
    if missing[layer_idx] != 1:
        return backhaul_usage
    
    cache = env.mec_cache.policy.cache

    if (-1, -1) in cache:
        action_idx = cache.index((-1, -1))
        message = {
            "video": req_state["video"],
            "tiles": [] if tile_idx is None else [req_state["viewport"][tile_idx]],
            "base_req_init": is_base,
            "action_idx": action_idx + 1,
        }
        env.prefetch_fn(env.mec_cache, message)

        backhaul_usage += bytes_cost
        missing[layer_idx] = 0

        return backhaul_usage

    if tile_idx is None:
        state = net_adapter.build_observation(0, req_state["video"])
        tiles = []
    else:
        tile = req_state["viewport"][tile_idx]
        state = net_adapter.build_observation(tile_idx, req_state["video"], tile)
        tiles = [tile]

    selection = agent.select_action(state)

    action = selection[0]
    value = selection[1] if len(selection) > 1 else None
    prob = selection[2] if len(selection) > 2 else None
    entropy = selection[3] if len(selection) > 3 else None

    if not is_base:
        pos_base_layer = net_adapter.env.mec_cache.policy.cache.index(
            (req_state["video"], -1)
        )

        if action == pos_base_layer + 1:
            return backhaul_usage

    transition[layer_idx] = {
        'state': state,
        'action': action,
        'value': value,
        'prob': prob,
        'entropy': entropy
    }

    message = {"video": req_state["video"], "tiles": tiles, "base_req_init": is_base, "action_idx": action}
    env.prefetch_fn(env.mec_cache, message)

    if action != 0:
        backhaul_usage += bytes_cost

    return backhaul_usage

def select_action(agent, req_state, env, net_adapter=None):

    if req_state is None:
        return None, np.zeros(5, dtype=np.int32), [None] * 5

    backhaul_usage = 0
    transition = [None] * 5

    missing = env._missing_items(req_state)

    backhaul_usage = process_missing_layer(
        agent=agent,
        req_state=req_state,
        env=env,
        net_adapter=net_adapter,
        missing=missing,
        transition=transition,
        backhaul_usage=backhaul_usage,
        layer_idx=0,
        tile_idx=None,
        is_base=True,
        bytes_cost=12 * env.mec_cache.tile_size_bytes[0],
    )

    has_base_layer = (req_state["video"], -1) in net_adapter.env.mec_cache.policy.cache

    if has_base_layer:
        for idx in range(cfg.viewport):
            backhaul_usage = process_missing_layer(
                agent=agent,
                req_state=req_state,
                env=env,
                net_adapter=net_adapter,
                missing=missing,
                transition=transition,
                backhaul_usage=backhaul_usage,
                layer_idx=idx + 1,
                tile_idx=idx,
                is_base=False,
                bytes_cost=env.mec_cache.tile_size_bytes[1],
            )

    log_selection_debug(transition, missing, backhaul_usage)

    return None, missing, transition

def run_episode(episode, env, agent, net_adapter, cfg, metrics_dir, global_step_start):
    """Run one full training episode and persist step-level metrics."""
    _, info = net_adapter.reset()

    total_reward = 0.0
    cache_hits = cache_misses = 0
    base_hits = base_misses = 0
    enh_hits = enh_misses = 0
    psnr_sum = 0.0

    episode_request_counts: dict[tuple[int, int], int] = defaultdict(int)

    global_step = global_step_start

    if cfg.has_warmup and episode == 0:
        env.warmup_phase(net_adapter, 1000)

    for step in range(cfg.max_steps):
        global_step += 1

        # --- Build State ---
        req_state = info.get("user_request", None)

        if req_state is not None:
            video_id = int(req_state["video"])
            episode_request_counts[_content_key(video_id, None)] += 1
            for tile in req_state.get("viewport", []):
                episode_request_counts[_content_key(video_id, int(tile))] += 1

        # --- Action Selection ---
        action, missing, transition = select_action(agent, req_state, env, net_adapter)

        # --- Environment Step ---
        _, reward, done, info = env.step(action, req_state, net_adapter)

        # --- Store Transition & Train ---
        nxt_req = info["user_request"]
        reward = info["reward_layer_0"] + info["reward_layer_1"]

        queued_update = False

        reward_sum = info["reward_layer_0"]
        if missing[0] == 1 or transition[0] is not None:
            state = net_adapter.build_observation(0, nxt_req["video"])
            agent.remember(
                transition[0]['state'],
                transition[0]['action'],
                reward_sum,
                state,
                done
            )
            queued_update = True

        for i in range(len(missing) - 1):
            reward_sum += info["reward_layer_1_details"][i] / cfg.viewport
            if missing[i + 1] == 1 and transition[i + 1] is not None:
                state = net_adapter.build_observation(i, nxt_req["video"], nxt_req["viewport"][i])
                agent.remember(
                    transition[i + 1]['state'],
                    transition[i + 1]['action'],
                    reward_sum,
                    state,
                    done
                )
                queued_update = True

        if queued_update:
            train_metrics = agent.train_step()

            if train_metrics is not None:
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

        debugger.log('cache_hits', cache_hits)
        debugger.log('cache_misses', cache_misses)

    cache_snapshot = _snapshot_cache(env)

    return (
        total_reward,
        cache_hits,
        cache_misses,
        base_hits,
        base_misses,
        enh_hits,
        enh_misses,
        global_step,
        psnr_sum / (step + 1),
        dict(episode_request_counts),
        cache_snapshot,
    )

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
    print(agent)

    global_log_path = os.path.join(cfg.path_results, "global.log")
    with open(global_log_path, "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now().isoformat()} | Max Steps: {cfg.max_steps} | ")
        f.write(f"{agent}\n")
        f.write("-" * 50 + "\n")

    global_step = 0

    for episode in range(cfg.n_episodes):
        (
            total_reward,
            cache_hits,
            cache_misses,
            base_hits,
            base_misses,
            enh_hits,
            enh_misses,
            global_step,
            psnr_rate,
            _episode_request_counts,
            _cache_snapshot,
        ) = run_episode(
            episode, env, agent, net_adapter, cfg, metrics_dir, global_step
        )
        
        agent.update_epsilon()

        # plot_topk_requested_with_cache(
        #     request_counts=episode_request_counts,
        #     cache_snapshot=cache_snapshot,
        #     episode=episode,
        #     top_k=top_k,
        #     out_dir=f"{debug_path}/plots",
        # )

        save_episode_metrics(
            metrics_dir=metrics_dir,
            ep=episode,
            total_reward=total_reward,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            agent=agent,
        )

        # save_episode_models(
        #     debug_path=debug_path, 
        #     episode=episode, 
        #     agent=agent
        # )

        debugger.save_results(filepath=f"{debug_path}/json/debug_ep{episode}")
        debugger.clear()

        base_hit_rate = _safe_hit_rate(base_hits, base_misses)
        enh_hit_rate = _safe_hit_rate(enh_hits, enh_misses)
        
        print(
            f"Episode {episode} | R: {int(total_reward)} | "
            f"HR: {_safe_hit_rate(cache_hits, cache_misses):.2f} | "
            f"HR_2: {(base_hit_rate + enh_hit_rate) / 2:.2f} | "
            f"BHR: {base_hit_rate:.2f} | "
            f"EHR: {enh_hit_rate:.2f} | "
            f"PSNR: {psnr_rate:.2f} | "
            f"Time: {pd.Timestamp.now().strftime('%H:%M:%S')}"
        )
        print("-" * 50)

if __name__ == "__main__":
    train(cfg)