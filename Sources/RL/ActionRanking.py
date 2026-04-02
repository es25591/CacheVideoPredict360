import numpy as np


class WolpertingerKnnSelector:
    """KNN selector that keeps action prototypes in learned embedding space.

    This is intentionally conservative: it preserves the model's policy probabilities,
    but restricts sampling to a small, diverse candidate set and applies a frequency
    penalty so a few actions do not dominate too early.
    """

    def __init__(
        self,
        action_dim: int,
        embedding_dim: int,
        k: int = 8,
        ema_alpha: float = 0.05,
        temperature: float = 1.0,
        frequency_penalty: float = 0.05,
        random_action_prob: float = 0.05,
    ):
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.action_dim = int(action_dim)
        self.embedding_dim = int(embedding_dim)
        self.k = int(max(1, min(k, action_dim)))
        self.ema_alpha = float(min(max(ema_alpha, 1e-4), 1.0))
        self.temperature = float(max(temperature, 1e-3))
        self.frequency_penalty = float(max(frequency_penalty, 0.0))
        self.random_action_prob = float(min(max(random_action_prob, 0.0), 0.5))

        rng = np.random.default_rng()
        prototypes = rng.normal(size=(self.action_dim, self.embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
        self._prototypes = prototypes / norms
        self._seen = np.zeros(self.action_dim, dtype=np.int64)

    def _nearest_candidates(self, state_embedding: np.ndarray) -> np.ndarray:
        deltas = self._prototypes - state_embedding.reshape(1, -1)
        distances = np.linalg.norm(deltas, axis=1)
        return np.argsort(distances)[: self.k]

    @staticmethod
    def _safe_softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        total = float(exp_values.sum())
        if total <= 0.0:
            return np.ones_like(exp_values) / max(len(exp_values), 1)
        return exp_values / total

    def select(self, state_embedding: np.ndarray, probs: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """Return selected action, candidate ids, and candidate probabilities."""
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        state_embedding = np.asarray(state_embedding, dtype=np.float32).reshape(-1)

        if probs.shape[0] != self.action_dim:
            raise ValueError("Probability vector size does not match action_dim")
        if state_embedding.shape[0] != self.embedding_dim:
            raise ValueError("Embedding size does not match embedding_dim")

        safe_probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(safe_probs.sum())
        if total <= 0.0:
            candidates = np.arange(self.action_dim, dtype=np.int64)
            return int(np.random.randint(0, self.action_dim)), candidates, np.ones_like(candidates, dtype=np.float32) / len(candidates)

        candidate_ids = self._nearest_candidates(state_embedding)
        candidate_probs = safe_probs[candidate_ids]

        # Temperature and frequency penalty reduce early action collapse.
        candidate_logits = np.log(candidate_probs + 1e-12) / self.temperature
        candidate_logits -= self.frequency_penalty * np.log1p(self._seen[candidate_ids].astype(np.float32))

        if self.random_action_prob > 0.0:
            candidate_distribution = self._safe_softmax(candidate_logits)
            uniform = np.ones_like(candidate_distribution) / len(candidate_distribution)
            candidate_distribution = (1.0 - self.random_action_prob) * candidate_distribution + self.random_action_prob * uniform
        else:
            candidate_distribution = self._safe_softmax(candidate_logits)

        if float(candidate_distribution.sum()) <= 0.0:
            action = int(candidate_ids[0])
        else:
            action = int(np.random.choice(candidate_ids, p=candidate_distribution))

        return action, candidate_ids, candidate_distribution

    def update(self, action_id: int, state_embedding: np.ndarray) -> None:
        state_embedding = np.asarray(state_embedding, dtype=np.float32).reshape(-1)
        if state_embedding.shape[0] != self.embedding_dim:
            return

        if not (0 <= int(action_id) < self.action_dim):
            return

        action_id = int(action_id)
        self._seen[action_id] += 1
        update_rate = self.ema_alpha / np.sqrt(float(self._seen[action_id]))
        self._prototypes[action_id] = (1.0 - update_rate) * self._prototypes[action_id] + update_rate * state_embedding

        norm = np.linalg.norm(self._prototypes[action_id]) + 1e-8
        self._prototypes[action_id] = self._prototypes[action_id] / norm
