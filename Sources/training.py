#!/usr/bin/env python
# coding: utf-8
"""
Refactored training entrypoint with improved readability, safety, and maintainability.

Key changes:
- Clean imports and typing
- Docstrings and logging
- Centralized path/env setup
- Safer Ray Tune lifecycle
- Removed broken DrlPolicy.get()
"""

from __future__ import annotations

import csv
import importlib
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import Checkpoint, run, sample_from  # noqa: F401 (kept for compatibility)
from ray.tune.progress_reporter import CLIReporter
from ray.tune.schedulers.pb2 import PopulationBasedTraining
from tensorboardX import SummaryWriter

# --------------------------------------------------------------------------------------
# Paths & Environment
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getcwd()).resolve().parent
SOURCES_ROOT = PROJECT_ROOT / "Sources"
PYTHONPATH_VALUE = os.pathsep.join([str(PROJECT_ROOT), str(SOURCES_ROOT)])

RAY_EXCLUDES: List[str] = [
    ".git/",
    ".venv/",
    "Dataset/",
    "Results/",
    "SampleVideos/",
    "Sources/Tests/deepmimo_scenarios/",
    "Sources/Tests/deepmimo_scenarios/*.zip",
]


def _ensure_sys_path(paths: List[Path]) -> None:
    """Prepend needed paths to sys.path, preserving existing entries."""
    for p in map(str, paths):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_pythonpath(value: str) -> None:
    """Ensure PYTHONPATH includes our composite path."""
    existing = os.environ.get("PYTHONPATH", "")
    parts = existing.split(os.pathsep) if existing else []
    if value not in parts:
        os.environ["PYTHONPATH"] = f"{value}{os.pathsep}{existing}" if existing else value


_ensure_sys_path([PROJECT_ROOT, SOURCES_ROOT])
_ensure_pythonpath(PYTHONPATH_VALUE)

# --------------------------------------------------------------------------------------
# External Modules (your project)
# --------------------------------------------------------------------------------------

import Common.config as config
import Common.datatypes as datatypes
import Common.debugger as debugger
import Common.utils as utils  # noqa: F401

# Hot-reload during development
importlib.reload(config)
importlib.reload(datatypes)
importlib.reload(debugger)
importlib.reload(utils)

# Typing aliases from your domain
UserTransition = datatypes.UserTransition  # noqa: F401
CachePolicy = datatypes.CachePolicy
CacheKey = datatypes.CacheKey  # noqa: F401

# Your debugger appears to expose an object (with .log/.save_results)
# named `debug` inside the module. Keep that instance.
debugger = debugger.debug

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

logger = logging.getLogger("training")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


class DrlPolicy(CachePolicy):
    """
    Minimal DRL caching policy with video/tile bookkeeping.

    Notes:
    - Removed broken `get()` (it referenced undefined `self.cache` and was unused here).
    - Methods align with likely CacheEngine expectations: put/contains/remove/clear/keys/capacity/stats.
    """

    EMPTY = -1

    def __init__(self, cfg: Any | None = None) -> None:
        self.cfg = cfg
        self.cur_size = 0
        self.video_idx: List[int] = [self.EMPTY] * self.cfg.cache_size
        self.tile_idx: List[List[int]] = [
            [self.EMPTY] * self.cfg.viewport for _ in range(self.cfg.cache_size)
        ]

    def put(self, key: int, value: Any, size: int) -> List[Any]:
        """
        Args:
            key: cache slot index to place the video.
            value: tuple(video_id, tiles)
            size: fixed as one slot (kept for interface compatibility)
        Returns:
            List of evicted items (empty for now).
        """
        evicted: List[Any] = []
        slot = key
        new_video, _ = value

        if new_video in self.video_idx:
            self.update_size()
            return evicted

        self.video_idx[slot] = new_video
        self.tile_idx[slot] = [self.EMPTY] * self.cfg.viewport

        self.update_size()
        return evicted

    def contains(self, key: CacheKey) -> bool:
        return key in self.video_idx

    def remove(self, key: CacheKey) -> bool:
        if key in self.video_idx:
            idx = self.video_idx.index(key)
            self.video_idx[idx] = self.EMPTY
            self.tile_idx[idx] = [self.EMPTY] * self.cfg.viewport
            self.update_size()
            return True
        return False

    def clear(self) -> None:
        self.video_idx = [self.EMPTY] * self.cfg.cache_size
        self.tile_idx = [[self.EMPTY] * self.cfg.viewport for _ in range(self.cfg.cache_size)]
        self.cur_size = 0

    def keys(self) -> List[int]:
        return self.video_idx

    def get_capacity(self) -> int:
        return self.cur_size

    def update_size(self) -> None:
        self.cur_size = sum(1 for v in self.video_idx if v != self.EMPTY)

    def stats(self) -> Dict[str, Any]:
        return {
            "current_size": self.cur_size,
            "capacity": self.cfg.cache_size,
            "num_items": sum(1 for v in self.video_idx if v != self.EMPTY),
        }


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def save_training_results(
    path_: str | Path,
    filename: str,
    ep: int,
    total_reward: float,
    cache_hits: int,
    cache_misses: int,
    agent: Any | None,
) -> None:
    """
    Append training metrics to a CSV file.

    Handles missing agent (None) gracefully.
    """
    filepath = Path(path_) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["episode", "total_reward", "cache_hits", "cache_misses", "epsilon", "lr"]

    write_header = not filepath.exists()

    with filepath.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        lr_val = None
        eps_val = None
        if agent is not None:
            try:
                lr_val = f"{agent.scheduler.get_last_lr()[0]:.10f}"
            except Exception:
                lr_val = None
            try:
                eps_val = round(float(agent.epsilon), 4)
            except Exception:
                eps_val = None

        writer.writerow(
            {
                "episode": int(ep),
                "total_reward": round(float(total_reward), 2),
                "cache_hits": int(cache_hits),
                "cache_misses": int(cache_misses),
                "lr": lr_val,
                "epsilon": eps_val,
            }
        )


# --------------------------------------------------------------------------------------
# Training Loop Entry (Ray Tune)
# --------------------------------------------------------------------------------------

# NOTE: TrainingManager is expected to be provided elsewhere in your project.
# It should expose: .train_fn() and agents with .update_epsilon(), etc.
# If you want, I can refactor that class next.

def train_rl_agent(trial_config: Dict[str, Any]) -> None:
    logger.info("RL training started with config: %s", trial_config)
    date_dir = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")

    cfg = config.Config()
    
    for key, value in trial_config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    cfg.filename =  (
        f"raytune_lrdecay{cfg.learning_rate}_batch{cfg.batch_size}_gamma{cfg.gamma}_optm{cfg.optimizer}_hid{cfg.hidden_dim}.csv"
    )

    debug_path = Path(cfg.path_results) / date_dir
    debug_path.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid circulars if any
    from Training.TrainingManager import TrainingManager

    manager = TrainingManager(cfg)
    tb_logger = SummaryWriter(log_dir=str(debug_path))

    try:
        for ep in range(cfg.n_episodes):
            (
                total_reward,
                hits,
                misses,
                bs_hits,
                bs_miss,
                enh_hits,
                enh_miss,
            ) = manager.train_fn()

            # Decay epsilons
            manager.base_agent.update_epsilon()
            manager.enh_agent.update_epsilon()

            # Persist to CSV
            save_training_results(
                path_=cfg.path_results,
                filename=cfg.filename,
                ep=ep,
                total_reward=total_reward,
                cache_hits=hits,
                cache_misses=misses,
                agent=manager.base_agent,
            )

            # Debugger metrics
            try:
                debugger.log("lr", manager.base_agent.scheduler.get_last_lr()[0])
                debugger.log("epsilon", manager.base_agent.epsilon)
                debugger.log("lr_enh", manager.enh_agent.scheduler.get_last_lr()[0])
                debugger.log("epsilon_enh", manager.enh_agent.epsilon)
            except Exception:
                pass

            # Hit-rate KPIs
            hr = hits / (hits + misses + 1e-9)
            bhr = bs_hits / (bs_hits + bs_miss + 1e-9)
            ehr = enh_hits / (enh_hits + enh_miss + 1e-9)

            # TensorBoard
            tb_logger.add_scalar("reward/total", total_reward, ep)
            tb_logger.add_scalar("cache/hit_rate", hr, ep) 
            tb_logger.add_scalar("cache/base_hit_rate", bhr, ep)
            tb_logger.add_scalar("cache/enh_hit_rate", ehr, ep)

            # Ray Tune report
            tune.report({
                "episode": int(ep),
                "total_reward": float(total_reward),
                "hit_rate": float(hr),
                "base_hit_rate": float(bhr),
                "enh_hit_rate": float(ehr),
            })

            # Persist debugger snapshot
            try:
                debugger.save_results(filepath=f"{debug_path}/debug_ep{ep}")
                debugger.clear()
            except Exception:
                pass

            logger.info(
                "Episode %d | R: %.2f | HR: %.3f | BHR: %.3f | EHR: %.3f",
                ep,
                total_reward,
                hr,
                bhr,
                ehr,
            )
    finally:
        tb_logger.close()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def _short_trial_dirname_creator(trial) -> str:
    return f"t_{trial.trial_id}"


def _short_trial_name_creator(trial) -> str:
    return f"trial_{trial.trial_id}"


def main() -> None:
    if ray.is_initialized():
        ray.shutdown()

    runtime_env = {
        "working_dir": str(PROJECT_ROOT),
        "excludes": RAY_EXCLUDES,
        "env_vars": {
            "PYTHONPATH": PYTHONPATH_VALUE,
            # Uncomment to force CPU:
            # "CUDA_VISIBLE_DEVICES": "",
        },
    }

    logger.info("Initializing Ray with runtime_env=%s", runtime_env)

    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    try:
        scheduler = PopulationBasedTraining(
            metric="total_reward",
            mode="max",
            perturbation_interval=160,
            hyperparam_mutations={
                "gamma": lambda: np.random.uniform(0.9, 0.999),
                "learning_rate": lambda: 10 ** np.random.uniform(-4, -1),
                "batch_size": [64, 128, 256],
                "optimizer": ["adam", "sgd"],
                "hidden_dims": [64, 128, 256],
            },
        )

        base_config = {
            "gamma": tune.uniform(0.9, 0.999),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([64, 128, 256]),
            "optimizer": tune.choice(["adam", "sgd"]),
            "hidden_dim": tune.choice([64, 128, 256]),
        }

        # Keep local path short to avoid TensorBoard file issues on Windows
        tune_storage_path = PROJECT_ROOT / "ray_runs"
        tune_storage_path.mkdir(parents=True, exist_ok=True)

        reporter = CLIReporter(
            max_progress_rows=10,
            max_report_frequency=30,
            print_intermediate_tables=True,
            metric_columns=["episode", "total_reward", "hit_rate", "base_hit_rate", "enh_hit_rate"],
        )

        logger.info("== Starting Ray Tune with Population Based Training scheduler ==")

        for seed in range(0, 4):
            run_config = dict(base_config)
            run_config["seed"] = seed

            analysis = tune.run(
                train_rl_agent,
                scheduler=scheduler,
                num_samples=10,
                reuse_actors=True,
                config=run_config,
                name=f"mmsp_pbt_s{seed}",
                storage_path=str(tune_storage_path),
                trial_dirname_creator=_short_trial_dirname_creator,
                trial_name_creator=_short_trial_name_creator,
                verbose=1,
                progress_reporter=reporter,
            )

            # Export results
            all_dfs = analysis.trial_dataframes
            names = list(all_dfs.keys())

            results = pd.DataFrame()
            for i in range(min(4, len(names))):
                df = all_dfs[names[i]].copy()
                df["sample_num"] = i
                results = pd.concat([results, df]).reset_index(drop=True)

            out_dir = Path("~/data/ray_tune_file_size4_method_env_default_max_160_batch").expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"seed{seed}.csv"
            results.to_csv(out_file, index=False)
            logger.info("Saved Ray Tune results to %s", out_file)

    finally:
        logger.info("Shutting down Ray.")
        ray.shutdown()


if __name__ == "__main__":
    main()
