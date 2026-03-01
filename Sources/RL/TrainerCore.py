from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


@dataclass
class EpisodeResult:
    state: Any
    last_action: Any
    last_hidden: Any
    transitions: list
    stats: Dict[str, float]
    info: Dict[str, Any]


def _to_bool_done(done: Any) -> bool:
    if isinstance(done, (bool, np.bool_)):
        return bool(done)
    if torch.is_tensor(done):
        if done.numel() == 0:
            return False
        return bool(torch.any(done).item())
    if isinstance(done, np.ndarray):
        if done.size == 0:
            return False
        return bool(np.any(done))
    if isinstance(done, Iterable) and not isinstance(done, (str, bytes, dict)):
        return any(bool(x) for x in done)
    return bool(done)


def _to_numpy_action(action: Any) -> Any:
    if torch.is_tensor(action):
        return action.detach().cpu().numpy()
    return action


def _safe_mean(value: Any) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.float().mean().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.astype(np.float32).mean())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.asarray(value, dtype=np.float32).mean())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_reset_output(reset_output: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(reset_output, tuple):
        if len(reset_output) == 2 and isinstance(reset_output[1], dict):
            return reset_output[0], reset_output[1]
        if len(reset_output) >= 1:
            return reset_output[0], {}
    return reset_output, {}


def normalize_step_output(step_output: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    if isinstance(step_output, tuple):
        if len(step_output) == 5:
            next_state, reward, terminated, truncated, info = step_output
            done = _to_bool_done(terminated) or _to_bool_done(truncated)
            return next_state, reward, done, info if isinstance(info, dict) else {}
        if len(step_output) == 4:
            next_state, reward, done, info = step_output
            return next_state, reward, _to_bool_done(done), info if isinstance(info, dict) else {}

    raise ValueError("Unsupported env.step output format. Expected 4 or 5 values.")

class EpisodeRunner:
    def __init__(self, env: Any, max_steps: int, cfg: Any, hyperparams: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.env = env
        self.max_steps = max_steps
        self.hyperparams = hyperparams

    def run(
        self,
        episode_i: int,
        stat: Dict[str, Any],
        schedule: Any,
        action_selector: Callable[..., Any],
        transition_handler: Callable[[Tuple[Any, Any, Any, Any, Any], int, int], None],
        last_hidden: Any = None,
        last_action: Any = None,
    ) -> EpisodeResult:
        stats = {
            "mean_train_reward": 0.0,
            "mean_train_solver_infeasible": 0.0,
            "mean_train_solver_interventions": 0.0,
        }
        transitions = []

        state, info = normalize_reset_output(self.env.reset())
        step_count = 0

        for step in range(self.max_steps):
            action = action_selector(
                state,
                schedule=schedule,
                last_act=last_action,
                last_hid=last_hidden,
                info=info,
                stat=stat,
            )

            next_state, reward, done, info = normalize_step_output(
                self.env.step(_to_numpy_action(action))
            )

            transition = (state, action, reward, next_state, done)
            transition_handler(transition, episode_i, step)
            transitions.append(transition)

            stats["mean_train_reward"] += _safe_mean(reward)
            stats["mean_train_solver_infeasible"] += _safe_mean(
                info.get("solver_infeasible") if isinstance(info, dict) else None
            )
            stats["mean_train_solver_interventions"] += _safe_mean(
                info.get("solver_interventions") if isinstance(info, dict) else None
            )

            state = next_state
            last_action = action
            last_hidden = None
            step_count += 1

            if done:
                break

        if step_count > 0:
            stats["mean_train_reward"] /= step_count
            stats["mean_train_solver_infeasible"] /= step_count
            stats["mean_train_solver_interventions"] /= step_count

        return EpisodeResult(
            state=state,
            last_action=last_action,
            last_hidden=last_hidden,
            transitions=transitions,
            stats=stats,
            info=info if isinstance(info, dict) else {},
        )
