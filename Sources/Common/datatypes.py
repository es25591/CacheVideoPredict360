
"""
Core domain models and policy interfaces for CacheVideoPredict360.

Design pattern:
- Strategy Pattern for cache policies (CachePolicy).
- Domain Model for request/tile representations.
- Factory Helper for ZipfSampler.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import numpy as np



# ================================================================
# 1. DOMAIN TYPES
# ================================================================

CacheKey = Tuple[int, int, int, int]   # (video_id, layer_id, tile_id, gop_id)


@dataclass(frozen=True)
class TileID:
    """A typed version of a tile identifier."""
    vid: int
    layer: int
    tile: int

    def as_key(self) -> CacheKey:
        return (self.vid, self.layer, self.tile)


@dataclass
class UserTransition:
    """Represents an RL transition for a user."""
    state: Any
    action: int
    reward: float


@dataclass
class UserRequest:
    """Represents one user's video-tile request."""
    user_id: int
    du_id: int
    tiles: List[TileID]


# ================================================================
# 2. CACHE POLICY INTERFACE  (Strategy Pattern)
# ================================================================

class CachePolicy(ABC):
    """
    Strategy interface for cache replacement policies.
    Implementations: LRU, LFU, Hybrid, Multi-Video LRU, etc.
    """

    @abstractmethod
    def get(self, key: CacheKey) -> Any:
        """Retrieve and mark as used."""
        ...

    @abstractmethod
    def put(self, key: CacheKey, value: Any, size: int):
        """Insert an item, potentially evicting others."""
        ...

    @abstractmethod
    def contains(self, key: CacheKey) -> bool:
        ...

    @abstractmethod
    def remove(self, key: CacheKey) -> bool:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def keys(self):
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        ...


# ================================================================
# 3. ZIPF SAMPLER (Factory Pattern)
# ================================================================

class ZipfSampler:
    """
    Factory-style sampler that generates video IDs following a Zipf distribution.
    """

    def __init__(
        self, 
        total_videos: int = 10, 
        alpha: float = 1.0, 
        seed: int | None = None
    ):
        self.total_videos = total_videos
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        weights = np.array(
            [1.0 / (i ** self.alpha) for i in range(1, self.total_videos + 1)],
            dtype=float
        )
        self.p = weights / weights.sum()

        self.videos = np.arange(1, total_videos + 1)

    def __call__(self) -> int:
        """Return a sampled video ID (1-indexed)."""
        idx = self.rng.choice(self.total_videos, p=self.p)
        return int(self.videos[idx])

    def sample_n(self, n: int) -> List[int]:
        """Convenience method for batch sampling."""
        idxs = self.rng.choice(self.total_videos, size=n, p=self.p)
        return [int(self.videos[i]) for i in idxs]


# ================================================================
# 4. TESTING HOOK
# ================================================================

if __name__ == "__main__":
    sampler = ZipfSampler(total_videos=100, alpha=1.0)
    samples = sampler.sample_n(500)
    
    print("Zipf Samples:", samples)
