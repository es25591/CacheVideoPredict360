from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


# --- DataClass for User Transition ---
@dataclass
class UserTransition:
    state: Any
    action: int
    reward: float

class ZipfSampler:
    def __init__(self, total_videos=10, alpha=1.0, seed=None):
        self.total_videos = total_videos
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        zipf_dist = [1.0 / (i ** alpha) for i in range(1, total_videos + 1)]
        s = sum(zipf_dist)
        self.zipf_dist = [x / s for x in zipf_dist]
        self.data = np.arange(1, total_videos + 1)

    def __call__(self):
        idx = self.rng.choice(self.total_videos, p=self.zipf_dist)
        return int(self.data[idx])

if __name__ == "__main__":
    total_videos = 100
    alpha = 1.0
    seed = None
    
    sampler = ZipfSampler(total_videos=total_videos, alpha=alpha, seed=seed)
    zipf_samples = [sampler() for _ in range(500)]
    print("Zipf Samples (500):", zipf_samples)

    counter = Counter(zipf_samples)
    print("Sample Counts:", counter)
