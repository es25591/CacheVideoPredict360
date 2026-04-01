import numpy as np


class WolpertingerKnnSelector:
    """KNN candidate selector for discrete actions using EMA-updated action prototypes."""

    def __init__(self, action_dim: int, k: int = 8, ema_alpha: float = 0.1):
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.action_dim = int(action_dim)
        self.k = int(max(1, min(k, action_dim)))
        self.ema_alpha = float(min(max(ema_alpha, 1e-4), 1.0))

        # Prototype space is policy-probability space with the same dimensionality as action space.
        self._prototypes = np.eye(self.action_dim, dtype=np.float32)
        self._seen = np.zeros(self.action_dim, dtype=np.int64)

    def _nearest_candidates(self, proto_action: np.ndarray) -> np.ndarray:
        deltas = self._prototypes - proto_action.reshape(1, -1)
        distances = np.linalg.norm(deltas, axis=1)
        return np.argsort(distances)[: self.k]

    def select(self, probs: np.ndarray) -> tuple[int, np.ndarray]:
        """Return selected action and candidate action ids."""
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        if probs.shape[0] != self.action_dim:
            raise ValueError("Probability vector size does not match action_dim")

        safe_probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(safe_probs.sum())

        if total <= 0.0:
            # Cold-start fallback.
            return int(np.random.randint(0, self.action_dim)), np.arange(self.action_dim, dtype=np.int64)

        proto_action = safe_probs / total
        candidates = self._nearest_candidates(proto_action)

        candidate_probs = proto_action[candidates]
        cand_total = float(candidate_probs.sum())
        if cand_total <= 0.0:
            action = int(candidates[0])
        else:
            candidate_probs = candidate_probs / cand_total
            action = int(np.random.choice(candidates, p=candidate_probs))

        # Online prototype update for the selected action.
        self._seen[action] += 1
        self._prototypes[action] = (1.0 - self.ema_alpha) * self._prototypes[action] + self.ema_alpha * proto_action

        return action, candidates
