
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
from enum import Enum, IntEnum
from typing import Any, Dict, Tuple, List
import numpy as np



# ================================================================
# 1. DOMAIN TYPES
# ================================================================

CacheKey = Tuple[int, int, int, int]   # (video_id, layer_id, tile_id, gop_id)

class LayerType(IntEnum):
    BASE = 0
    ENHANCEMENT = 1

@dataclass(frozen=True)
class TileID:
    """
    Strongly typed representation of a tile in a video.
    Each tile has:
        - vid: video identifier
        - layer: quality layer (0 = base layer, >0 = enhancement)
        - tile: tile index within a GOP or frame
    """
    vid: int
    layer: int
    tile: int

    def as_key(self, gop_id: int = 0) -> CacheKey:
        """
        Convert tile information into a CacheKey.
        GOP is optional and defaults to 0 unless provided by the cache.
        """
        return (self.vid, self.layer, self.tile, gop_id)


@dataclass
class UserTransition:
    """
    Represents a single RL transition for a specific user.

    Attributes:
        state:  Observation before action
        action: The integer-coded action taken by the agent
        reward: Reward obtained after transition
    """
    state: Any
    next_state: Any
    action: int
    reward: float


@dataclass
class TransitionTuple:
    """
    Represents a single RL transition tuple (state, action, reward, next_state).
    Attributes:
        state:  Observation before action
        action: The integer-coded action taken by the agent
        reward: Reward obtained after transition
    """
    state: Any
    next_state: Any
    action: int
    reward: float
    done: bool = False


@dataclass
class UserRequest:
    """
    A request issued by a user for a group of tiles.

    Attributes:
        user_id:  Unique user ID
        du_id:    Index of the Delivery Unit (cache node) serving that user
        tiles:    A list of TileID objects requested
    """
    user_id: int
    du_id: int
    tiles: List[TileID]


# ================================================================
# 2. VIDEO CATEGORY & WEIBULL PARAMETERS
# ================================================================

class VideoCategory(Enum):
    """
    Categories of videos used for popularity modeling.
    Using Enum provides stronger typing and prevents accidental mismatches.
    """
    GAMING = "Gaming"
    COMEDY = "Comedy"
    ENTERTAINMENT = "Entertainment"
    EDUCATION = "Education"
    NEWS = "News"
    SCIENCE = "Science"
    MUSIC = "Music"
    AUTOS = "Autos"
    SPORTS = "Sports"
    FILM = "Film"


@dataclass(frozen=True)
class WeibullParams:
    """
    Parameters defining a 3-parameter Weibull distribution.

    Attributes:
        alpha: shape parameter (>0)
        beta:  scale parameter (>0)
        gamma: shift parameter
    """
    alpha: float
    beta: float
    gamma: float


# A mapping of category to its Weibull parameters.
CATEGORY_WEIBULL: Dict[VideoCategory, WeibullParams] = {
    VideoCategory.GAMING:        WeibullParams(1.98, 0.45,  0.0146),
    VideoCategory.COMEDY:        WeibullParams(2.89, 0.65, -0.0250),
    VideoCategory.ENTERTAINMENT: WeibullParams(2.41, 0.56, -0.0064),
    VideoCategory.EDUCATION:     WeibullParams(2.40, 0.54, -0.0104),
    VideoCategory.NEWS:          WeibullParams(4.70, 0.95, -0.2980),
    VideoCategory.SCIENCE:       WeibullParams(2.53, 0.53,  0.0130),
    VideoCategory.MUSIC:         WeibullParams(2.45, 0.51,  0.0178),
    VideoCategory.AUTOS:         WeibullParams(2.68, 0.58,  0.0016),
    VideoCategory.SPORTS:        WeibullParams(4.34, 0.92, -0.2670),
    VideoCategory.FILM:          WeibullParams(2.32, 0.62,  0.0205),
}


# ================================================================
# 2. CACHE POLICY INTERFACE  (Strategy Pattern)
# ================================================================

class CachePolicy(ABC):
    """
    Abstract Strategy interface for implementing cache replacement algorithms.

    Concrete implementations may include:
        - LRU (Least Recently Used)
        - LFU (Least Frequently Used)
        - Video-aware LRU
        - Two-layer policies for 360-degree tile-based streaming

    Methods:
        get(key):        retrieve value and update recency
        put(key, value): insert new data, applying eviction logic
        contains(key):   check presence
        remove(key):     explicit removal
        clear():         remove all entries
        keys():          iterator over stored keys
        stats():         diagnostics for debugging or analytics
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
    Sampler for choosing video IDs using a Zipf distribution.

    Usage:
        sampler = ZipfSampler(total_videos=100, alpha=1.2)
        v = sampler()            # sample 1 video
        batch = sampler.sample_n(20)
    """

    def __init__(
        self,
        total_videos: int = 10,
        alpha: float = 1.0,
        seed: int | None = None
    ):
        if total_videos <= 0:
            raise ValueError("total_videos must be positive.")

        self.total_videos = total_videos
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        raw_weights = np.array(
            [1.0 / (i ** alpha) for i in range(1, total_videos + 1)],
            dtype=float,
        )
        self.probabilities = raw_weights / raw_weights.sum()
        self.videos = np.arange(1, total_videos + 1)

    def __call__(self) -> int:
        """Sample a single video ID from Zipf distribution."""
        idx = self.rng.choice(self.total_videos, p=self.probabilities)
        return int(self.videos[idx])

    def sample_n(self, n: int) -> List[int]:
        """Sample n video IDs in batch form."""
        idxs = self.rng.choice(self.total_videos, size=n, p=self.probabilities)
        return [int(self.videos[i]) for i in idxs]


# ================================================================
# 4. TESTING HOOK
# ================================================================

if __name__ == "__main__":
    sampler = ZipfSampler(total_videos=500, alpha=0.8)
    samples = sampler.sample_n(200)
    
    print("Zipf Samples:", samples)

    counter = {vid: samples.count(vid) for vid in set(samples)}
    print("Sample Counts:", counter)

    first_50_count = sum(counter.get(vid, 0) for vid in range(1, 51))
    print(f"Total requests for first 50 videos: {first_50_count}")