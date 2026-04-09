import numpy as np

import Common.debugger as debugger


def build_latency_model(cfg):
    """Build and return the MultiDULatencyModel."""
    from importnb import Notebook
    with Notebook():
        from Labs.LatencyModel import MultiDULatencyModel

    P = cfg.n_nodes
    max_U = cfg.n_users

    return MultiDULatencyModel(
        P=P,
        max_U=max_U,
        R_M_D=80e6,
        R_C_M=125e6,
        mu=2e7,
        eta=2e5,
        B_pu_matrix=np.full((P, max_U), 20e6, dtype=float),
        gamma_pu_matrix=np.full((P, max_U), 5.0, dtype=float),
        rhoT_p=[0.2],
        lambda_p=[0.05],
        du_fixed_delay=0.001,
        mec_fixed_delay=0.005,
        cloud_fixed_delay=0.1
    )

def build_environment(cfg):
    """Construct the full multi-component environment wrapper."""
    from importnb import Notebook
    with Notebook():
        from Labs.Policy import DrlPolicy, MMSPPolicy
        from Labs.CacheEngine import CacheEngineEnv
        from Labs.UserRequest import UserRequestEvents

        if cfg.env_type == "cpt":
            from Labs.EnvCPTWrapper import EnvWrapper
        else:
            from Labs.EnvWrapper import EnvWrapper

    du_caches = []

    # DRL Caching Policy
    # policy = DrlPolicy(cfg=cfg)
    policy = MMSPPolicy(cfg=cfg)
    
    # MEC Cache Engine
    mec_cache = CacheEngineEnv(
        n_users=cfg.n_users,
        n_videos=cfg.n_videos,
        n_layers=cfg.n_layers,
        n_tiles=cfg.n_tiles,
        n_gops=cfg.n_gops,
        cache_capacity=cfg.cache_capacity,
        policy=policy
    )

    # User request generator
    users_env = UserRequestEvents(
        n_nodes=cfg.n_nodes,
        n_users=cfg.n_users,
        n_videos=cfg.n_videos,
        n_gops=cfg.n_gops,
        n_layers=cfg.n_layers,
        n_tiles=cfg.n_tiles,
        n=cfg.n,
        m=cfg.m,
        arrival_rate=cfg.arrival_rate,
        zipf_alpha=cfg.zipf_alpha
    )

    # Latency Model
    latency_model = build_latency_model(cfg)

    # Wrapping all into the main training environment
    return EnvWrapper(
        cfg=cfg,
        n=cfg.n,
        m=cfg.m,
        n_layers=cfg.n_layers,
        users_env=users_env,
        du_caches=du_caches,
        mec_cache=mec_cache,
        latency_model=latency_model,
        theta=cfg.theta,
        lam=cfg.lam,
        max_steps=cfg.max_steps,
        prefetch_fn=lambda cache, action: cache.drl_prefetching_focus(action),
        reward_fn=lambda env, reqs: env.compute_reward(reqs),
        debugger=debugger.debug
    )