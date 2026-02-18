# --- 1. CONFIGURATION & HYPERPARAMETERS (Section VII-B) ---
from dataclasses import dataclass

class Config:
    n_episodes: int = 300
    max_steps: int = 10000
    n_nodes: int = 3
    n_users: int = 200
    step_size: float = 10.0
    arrival_rate: float = 200.0  # users per second
    zipf_alpha: float = 0.8
    n_videos: int = 500
    n_gops: int = 30
    n_layers: int = 2
    n: int = 4
    m: int = 3
    n_tiles: int = n * m
    viewport: int = 4
    T_chunk_default: float = 1.0  # default chunk deadline in seconds

    bas_layer_size: float = 2.0e+6  # 2 MB
    enh_layer_size: float = 1.5e+7  # 15 MB
    total_video_size: float = n_videos * n_gops * (bas_layer_size  + viewport * (enh_layer_size / n_tiles))  # total size of all videos in bytes

    cache_capacity_percent: float = 0.1  # 10% of the total video size
    cache_capacity: float = cache_capacity_percent * total_video_size
    cache_size: int = int(cache_capacity_percent * n_videos)

    # Hyperparameters for RL
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.987 # (epsilon_min / epsilon_start) ** (1.0 / n_episodes)
    entropy_coef: float = 0.02

    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.9999

    tau: float = 0.01
    gamma: float = 0.99 # 0.6
    batch_size: int = 32
    buffer_capacity: int = 2000
    nb_interval: int = 200  # interval to update target network

    h_short: int = 300   # sliding windows for popularity (Section VI-A)
    h_long: int = 1000

    # CPT parameters
    theta: float = 0.5
    lam: float = 3.7183

    # Paths for data and results

    # path_data: str = '/home/eduardo/Workspace/CacheVideoPredict360/Data'
    # path_results: str = '/home/eduardo/Workspace/CacheVideoPredict360/Results'
    # path_trajectories: str = '/home/eduardo/Workspace/CacheVideoPredict360/Dataset/Trajectories'

    path_data: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Data'
    path_results: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Results'
    path_trajectories: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Dataset\Trajectories'

    filename: str = f"drl_dqn_lrdecay{learning_rate_decay}_c{cache_size}_ar{arrival_rate}_z{zipf_alpha}.csv"

    state_dim: int = 10 * cache_size + 2 # 10*C + 2 = (2C + 2Ck) * 2 + 2 (Section VI-A)
    action_dim: int = 2  # 0 = Pass, 1 = Cache

    num_goals = cache_size + 1
    num_low_actions = 5

    state_dim_meta: int = 2 * cache_size + 2 
    state_dim_ctrl: int = 8 * cache_size + 8 
