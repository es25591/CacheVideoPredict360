# --- 1. CONFIGURATION & HYPERPARAMETERS (Section VII-B) ---
from dataclasses import dataclass

class Config:
    has_warmup: int = True

    env_type: str = "standard"  # "cpt" or "standard"

    n_episodes: int = 400
    max_steps: int = 20000
    n_nodes: int = 3
    n_users: int = 200
    step_size: float = 10.0
    arrival_rate: float = 200.0  # users per second
    zipf_alpha: float = 1.0
    n_videos: int = 500
    n_gops: int = 30
    user_session_length: int = 30
    n_layers: int = 2
    n: int = 4
    m: int = 3
    n_tiles: int = n * m
    viewport: int = 4
    T_chunk_default: float = 1.0  # default chunk deadline in seconds

    bas_layer_size: float = 2.0e+6  # 2 MB
    enh_layer_size: float = 1.5e+7  # 15 MB
    total_video_size: float = n_videos * n_gops * (bas_layer_size  + viewport * (enh_layer_size / n_tiles))  # total size of all videos in bytes

    cache_capacity_percent: float = 0.20  # 20% of the total video size
    cache_capacity: float = cache_capacity_percent * total_video_size
    cache_size: int = int(cache_capacity_percent * n_videos)

    # Hyperparameters for RL
    epsilon_start: float = 0.05
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.985
    base_exploration_strategy: str = "epsilon_greedy"  # "epsilon_greedy" or "boltzmann"
    temperature_start: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.995
    entropy_coef: float = 0.02

    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.9995
    learning_rate_actor: float = 1e-3
    learning_rate_critic: float = 2.5e-3

    tau: float = 0.01
    gamma: float = 0.6 # 0.6
    batch_size: int = 32
    buffer_capacity: int = 2000
    nb_interval: int = 200  # interval to update target network
    n_step: int = 1000
    optimizer: str = "adam"  # "adam" or "sgd"

    # A2C-specific hyperparameters
    gae_lambda: float = 0.95  # GAE lambda for variance reduction
    entropy_beta: float = 0.01  # Entropy regularization coefficient
    advantage_clip: float = 5.0  # Clip advantages to [-clip, clip]
    gradient_clip_norm: float = 0.5  # Max gradient norm clipping
    action_temperature: float = 1.2
    random_action_prob: float = 0.05

    # Wolpertinger (KNN candidate pruning) parameters
    wolpertinger_enabled: bool = False
    wolpertinger_k_base: int = 8
    wolpertinger_k_enh: int = 4
    wolpertinger_ema_alpha: float = 0.1
    wolpertinger_temperature: float = 1.2
    wolpertinger_frequency_penalty: float = 0.05
    wolpertinger_random_action_prob: float = 0.05

    h_short: int = 300   # sliding windows for popularity (Section VI-A)
    h_long: int = 1000

    # CPT parameters
    theta: float = 0.5
    lam: float = 3.7183

    # Paths for data and results

    ### Local Laptop
    path_data: str = '/home/eduardo/Workspace/CacheVideoPredict360/Data'
    path_results: str = '/home/eduardo/Workspace/CacheVideoPredict360/Results'
    path_trajectories: str = '/home/eduardo/Workspace/CacheVideoPredict360/Dataset/Trajectories'

    ### CERES
    # path_data: str = '/home/es25591/CacheVideoPredict360/Data'
    # path_results: str = '/home/es25591/CacheVideoPredict360/Results'
    # path_trajectories: str = '/home/es25591/CacheVideoPredict360/Dataset/Trajectories'

    ### Local 6G Lab
    # path_data: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Data'
    # path_results: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Results'
    # path_trajectories: str = r'c:\Users\es25591\Workspace\CacheVideoPredict360\Dataset' + r'\GeneratedTrajectories\len60'

    filename: str = f"lrdecay{learning_rate_decay}_c{cache_size}_ar{arrival_rate}_z{zipf_alpha}_gamma{gamma}.csv"

    omega: float = 1.0 # weight for heuristic values in action selection

    state_dim: int = 10 * cache_size + 2 # 10*C + 2 = (2C + 2Ck) * 2 + 2 (Section VI-A)
    action_dim: int = 2  # 0 = Pass, 1 = Cache
    hidden_dim: int = 128
    hidden_dims: tuple = (512,)
    
    action_dim_base_focus: int = cache_size + 1
    action_dim_enh_focus: int = viewport + 1

    state_dim_base_focus: int = 2 * cache_size + 2 
    state_dim_enh_focus: int = 2 * viewport + 2 * viewport

    hidden_dim_base_focus: int = 512
    hidden_dim_enh_focus: int = 128
    hidden_dim_focus: int = 1024
    hidden_dims_focus: tuple = (3012, 1024, 512)

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