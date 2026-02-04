import csv
import os
import numpy as np

from collections import Counter

def get_required_tiles(center_yaw, center_pitch, n=12, fov_yaw=90.0, fov_pitch=90.0):
    """
    Given a center yaw and pitch, compute the required tiles.

    Parameters:
    - center_yaw: Center yaw angle in degrees (0-360).
    - center_pitch: Center pitch angle in degrees (-90 to +90).
    - n: Number of tiles per axis (e.g., 8x8 grid).
    - fov_yaw: Field of view in yaw direction in degrees.
    - fov_pitch: Field of view in pitch direction in degrees.

    Returns:
    - A list of (tile_x, tile_y) tuples representing required tiles.
    """
    tile_size_yaw = 360 / n
    tile_size_pitch = 180 / n  # Pitch ranges from -90 to +90

    # Calculate the FOV boundaries
    half_fov_yaw = fov_yaw / 2
    half_fov_pitch = fov_pitch / 2

    # Determine the min and max yaw and pitch
    min_yaw = (center_yaw - half_fov_yaw) % 360
    max_yaw = (center_yaw + half_fov_yaw) % 360
    min_pitch = max(center_pitch - half_fov_pitch, -90)
    max_pitch = min(center_pitch + half_fov_pitch, 90)

    required_tiles = set()

    # Calculate tile indices for yaw
    if min_yaw < max_yaw:
        yaw_indices = range(int(min_yaw // tile_size_yaw), int(max_yaw // tile_size_yaw) + 1)
    else:
        yaw_indices = list(range(int(min_yaw // tile_size_yaw), n)) + list(range(0, int(max_yaw // tile_size_yaw) + 1))

    # Calculate tile indices for pitch
    pitch_indices = range(int((min_pitch + 90) // tile_size_pitch), int((max_pitch + 90) // tile_size_pitch) + 1)

    # Combine yaw and pitch indices to get required tiles
    for yaw_index in yaw_indices:
        for pitch_index in pitch_indices:
            required_tiles.add((yaw_index % n, pitch_index))

    return list(required_tiles)

def poisson_per_time(total_time, rate_per_minute):
    lam = rate_per_minute / 60

    # Calculate the total number of events
    total_events = int(total_time * rate_per_minute)

    # Generate inter-event times
    inter_event_times = np.random.exponential(1/lam, total_events)

    return inter_event_times

def poisson_global_times(total_time, rate_per_minute):

    # Convert rate to events per second
    lam = rate_per_minute / 60.0

    # Calculate the expected total number of events
    total_events = int(total_time * rate_per_minute)

    # Generate inter-event times (in seconds) scale = 1/lambda
    inter_event_times = np.random.exponential(scale=1.0/lam, size=total_events)

    # Calculate global times by cumulatively summing inter-event times
    global_times = np.cumsum(inter_event_times)
    
    return np.round(global_times).astype(int)

def poisson_per_users(total_users, rate_per_minute):
    lam = rate_per_minute / 60.0

    if total_users <= 0:
        return np.array([], dtype=int)

    inter_event_times = np.random.exponential(scale=1.0/lam, size=total_users)

    global_times = np.cumsum(inter_event_times)

    return np.round(global_times).astype(int)

def poisson_per_video_requests(total_requests, rate_per_minute):
    lam = rate_per_minute / 60.0

    if total_requests <= 0:
        return np.array([], dtype=int)

    inter_event_times = np.random.exponential(scale=1.0/lam, size=total_requests)

    global_times = np.cumsum(inter_event_times)

    return global_times



def zipf(samples = None, total_videos = 10, alpha = 1.0):
    zipf_dist = [1.0 / (i ** alpha) for i in range(1, total_videos + 1)]
    zipf_dist = [x / sum(zipf_dist) for x in zipf_dist]

    indices = np.random.choice(total_videos, samples, p=zipf_dist)
    data = np.random.permutation(np.arange(1, total_videos + 1))

    elements = [int(data[i]) for i in indices]

    return elements

def save_training_results(
    path_,
    filename,
    ep, 
    total_reward, 
    cache_hits, 
    cache_misses, 
    enhanced_layer_cache_hits,
    enhanced_layer_cache_misses,
    base_layer_cache_hits,
    base_layer_cache_misses,
    avg_psnr,
    agent
):
    with open(os.path.join(path_, filename), 'a', newline='') as f:
        fieldnames = [
            'episode', 
            'total_reward', 
            'cache_hits', 
            'cache_misses', 
            'enhanced_layer_cache_hits', 
            'enhanced_layer_cache_misses',
            'base_layer_cache_hits', 
            'base_layer_cache_misses', 
            'average_psnr',
            'epsilon'
        ]
        writer_results = csv.DictWriter(f, fieldnames=fieldnames)

        if ep == 0:
            writer_results.writeheader()
        
        writer_results.writerow({
            'episode': ep,
            'total_reward': round(float(total_reward), 2),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'enhanced_layer_cache_hits': enhanced_layer_cache_hits,
            'enhanced_layer_cache_misses': enhanced_layer_cache_misses,
            'base_layer_cache_hits': base_layer_cache_hits,
            'base_layer_cache_misses': base_layer_cache_misses,
            'average_psnr': round(float(avg_psnr), 2),
            'epsilon': round(float(agent.epsilon), 2) if agent else None
        })

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

def zipf_sampler(total_videos=10, alpha=1.0, seed=None):
    return ZipfSampler(total_videos=total_videos, alpha=alpha, seed=seed)


if __name__ == "__main__":
    # Example usage
    center_yaw = 110
    center_pitch = 50

    required_tiles = get_required_tiles(
        center_yaw, center_pitch, n=8, fov_yaw=90, fov_pitch=90
    )

    # print("Required Tiles:", required_tiles)

    zipf_samples = zipf(samples=200, total_videos=10, alpha=1.0)
    print("Zipf Samples:", zipf_samples)

    counter = Counter(zipf_samples)
    print("Sample Counts:", counter)

    # zipf_samples = zipf(samples=100, total_videos=1000, alpha=2.0)
    # print("Zipf Samples:", zipf_samples)

    probabilities = [x / sum(counter.values()) for x in counter.values()]

    chosen_probabilities = sorted(
        [(video_id, prob * 100) for video_id, prob in zip(counter.keys(), probabilities)],
        key=lambda x: x[1],
        reverse=True
    )

    # print("Video Choice Probabilities for Chosen Videos:")
    # for i, (video_id, prob) in enumerate(chosen_probabilities):
    #     print(f"  {i+1}. Video {video_id}: {prob:.1f}% (chosen {counter[video_id]} times)")

    # poisson_times = poisson_per_time(total_time=2, rate_per_minute=10)
    # print("Poisson Inter-Event Times:", poisson_times)

    # global_times = poisson_global_times(total_time=2, rate_per_minute=10)
    # print("Poisson Global Timestamps (sec):", global_times, len(global_times))

    # poisson_user_counts = poisson_per_users(total_users=10, rate_per_minute=5)
    # print("Poisson User Counts:", poisson_user_counts)

    sampler = zipf_sampler(total_videos=10, alpha=1.0, seed=None)
    zipf_samples = [sampler() for _ in range(500)]
    print("Zipf Samples (500):", zipf_samples)

    counter = Counter(zipf_samples)
    print("Sample Counts:", counter)
