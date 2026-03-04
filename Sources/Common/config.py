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
    user_session_length: int = 60
    n_layers: int = 2
    n: int = 4
    m: int = 3
    n_tiles: int = n * m
    viewport: int = 4
    T_chunk_default: float = 1.0  # default chunk deadline in seconds

    bas_layer_size: float = 2.0e+6  # 2 MB
    enh_layer_size: float = 1.5e+7  # 15 MB
    total_video_size: float = n_videos * n_gops * (bas_layer_size  + viewport * (enh_layer_size / n_tiles))  # total size of all videos in bytes

    cache_capacity_percent: float = 0.2  # 20% of the total video size
    cache_capacity: float = cache_capacity_percent * total_video_size
    cache_size: int = int(cache_capacity_percent * n_videos)

    # Hyperparameters for RL
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.985
    entropy_coef: float = 0.02

    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.9995

    tau: float = 0.01
    gamma: float = 0.9 # 0.6
    batch_size: int = 32
    buffer_capacity: int = 2000
    nb_interval: int = 200  # interval to update target network
    n_step: int = 1000
    optimizer: str = "adam"  # "adam" or "sgd"


    h_short: int = 300   # sliding windows for popularity (Section VI-A)
    h_long: int = 1000

    # CPT parameters
    theta: float = 0.5
    lam: float = 3.7183

    # Paths for data and results

    ### Local Laptop
    # path_data: str = '/home/eduardo/Workspace/CacheVideoPredict360/Data'
    # path_results: str = '/home/eduardo/Workspace/CacheVideoPredict360/Results'
    # path_trajectories: str = '/home/eduardo/Workspace/CacheVideoPredict360/Dataset/Trajectories'

    ### CERES
    # path_data: str = '/home/es25591/CacheVideoPredict360/Data'
    # path_results: str = '/home/es25591/CacheVideoPredict360/Results'
    # path_trajectories: str = '/home/es25591/CacheVideoPredict360/Dataset/Trajectories'

    ### Local 6G Lab
    path_data: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Data'
    path_results: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Results'
    path_trajectories: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Dataset\Trajectories'

    filename: str = f"lrdecay{learning_rate_decay}_c{cache_size}_ar{arrival_rate}_z{zipf_alpha}_gamma{gamma}.csv"

    state_dim: int = 10 * cache_size + 2 # 10*C + 2 = (2C + 2Ck) * 2 + 2 (Section VI-A)
    action_dim: int = 2  # 0 = Pass, 1 = Cache
    hidden_dim: int = 128

    action_dim_base_focus: int = cache_size + 1
    action_dim_enh_focus: int = viewport + 1

    state_dim_base_focus: int = 2 * cache_size + 2 
    state_dim_enh_focus: int = 2 * viewport + 2 * viewport

    hidden_dim_base_focus: int = 512
    hidden_dim_enh_focus: int = 128

    has_warmup: int = True
    ### Additional configuration parameters can be added here as needed

    init_type: str = "normal"  # "normal" or "orthogonal"
    init_std: float = 0.02

    n_agents: int = 1
    agent_idx: int = 0
    agent_type: str = "mlp"  # "mlp" or "transformer"
    
    hidden_size: int = 128
    hidden_dim: int = 128
    attend_heads: int = 4
    replay: bool = True
    shared_params: bool = True
    continuous: bool = False
    norm_in: bool = True
    layernorm: bool = False
    hid_activation: str = 'relu'  # 'relu' or 'tanh'
    
    LOG_STD_MIN: float = -20
    LOG_STD_MAX: float = 2

    def __dict__(self):
        return {k: getattr(self, k) for k in self.__annotations__.keys()}

    def __str__(self):
        return "\n".join(f"{k}: {getattr(self, k)}" for k in self.__annotations__.keys())

    