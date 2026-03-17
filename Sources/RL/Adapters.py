import numpy as np

from collections import defaultdict, deque
from typing import Dict, Any


class FeatureAdapter:
    def __init__(self, cfg: Any, env: Any):
        self.env = env
        self.cfg = cfg

        self.video_hist_short = deque(maxlen=cfg.h_short)
        self.video_hist_long = deque(maxlen=cfg.h_long)
        self.tile_hist_short = deque(maxlen=cfg.h_short * cfg.viewport)
        self.tile_hist_long = deque(maxlen=cfg.h_long * cfg.viewport)
        self.tiles_hist_short = deque(maxlen=cfg.h_short)
        self.tiles_hist_long = deque(maxlen=cfg.h_long)

        self.video_freq_short = defaultdict(int)
        self.video_freq_long = defaultdict(int)
        self.tile_freq_short = defaultdict(int)
        self.tile_freq_long = defaultdict(int)
        self.tiles_freq_short = defaultdict(int)
        self.tiles_freq_long = defaultdict(int)

        self.ch_video_hist = deque(maxlen=cfg.h_long)
        self.ch_viewport_hist = deque(maxlen=cfg.h_long)

        self.all_video_hist = []

    def reset_history(self):
        queues = (
            self.video_hist_short,
            self.video_hist_long,
            self.tiles_hist_short,
            self.tiles_hist_long,
            self.tile_hist_short,
            self.tile_hist_long,
            self.ch_video_hist,
            self.ch_viewport_hist,
        )
        freqs = (
            self.video_freq_short,
            self.video_freq_long,
            self.tiles_freq_short,
            self.tiles_freq_long,
            self.tile_freq_short,
            self.tile_freq_long,
        )

        for q in queues:
            q.clear()
        for f in freqs:
            f.clear()

        self.all_video_hist = []

    def update_history(self, vid: int, tiles: list[int]):       
        self._update_window(self.video_hist_short, self.video_freq_short, vid)
        self._update_window(self.video_hist_long, self.video_freq_long, vid)

        tiles = tuple(tiles) if tiles is not None else None

        if tiles is None:
            return

        self._update_window(self.tiles_hist_short, self.tiles_freq_short, tiles)
        self._update_window(self.tiles_hist_long, self.tiles_freq_long, tiles)

        for tile in tiles:
            self._update_window(self.tile_hist_short, self.tile_freq_short, (vid, tile))
            self._update_window(self.tile_hist_long, self.tile_freq_long, (vid, tile))

    def update_history_single(self, item):
        if isinstance(item, int):
            self._update_window(self.video_hist_short, self.video_freq_short, item)
            self._update_window(self.video_hist_long, self.video_freq_long, item)
        elif isinstance(item, tuple) and len(item) == 2:
            self._update_window(self.tile_hist_short, self.tile_freq_short, item)
            self._update_window(self.tile_hist_long, self.tile_freq_long, item)
        elif isinstance(item, tuple) and all(isinstance(i, int) for i in item):
            self._update_window(self.tiles_hist_short, self.tiles_freq_short, item)
            self._update_window(self.tiles_hist_long, self.tiles_freq_long, item)

    def update_ch_history(self, vid: int, viewport: list[int]) -> None:
        self.all_video_hist.append(vid)

        video_cache_index = self.env.mec_cache.policy.video_idx
        tile_cache_index = self.env.mec_cache.policy.tile_idx

        hit = 1 if vid in video_cache_index else 0
        self.ch_video_hist.append(hit)

        if not hit:
            viewport_vector = [0,0,0,0]
        else:
            idx = video_cache_index.index(vid)
            cached_tiles = tile_cache_index[idx]
            viewport_vector = [1 if tile in cached_tiles else 0 for tile in viewport]

        self.ch_viewport_hist.append(viewport_vector)

    def compute_reward_layer_0(self, window_size: int = None) -> float:
        if window_size is None:
            window_size = len(self.ch_video_hist)

        ch_video_list = list(self.ch_video_hist)[-window_size:]
        psnr_layer_0 = 30 * sum(ch_video_list)
        
        return psnr_layer_0 / len(ch_video_list) if ch_video_list else 0

    def compute_reward_layer_1(self, window_size: int = None) -> float:
        if window_size is None:
            window_size = len(self.ch_viewport_hist)

        ch_viewport_list = list(self.ch_viewport_hist)[-window_size:]
        psnr_layer_1 = 2.5 * sum(sum(viewport) for viewport in ch_viewport_list)

        return psnr_layer_1 / len(ch_viewport_list) if ch_viewport_list else 0

    def compute_reward(self, window_size: int = None) -> float:
        if window_size is None:
            window_size = len(self.ch_video_hist)
        
        ch_video_list = list(self.ch_video_hist)[-window_size:]
        ch_viewport_list = list(self.ch_viewport_hist)[-window_size:]

        psnr_layer_0 = 30 * sum(ch_video_list)
        psnr_layer_1 = 2.5 * sum(sum(viewport) for viewport in ch_viewport_list)

        total_items = len(ch_video_list)
        return (psnr_layer_0 + psnr_layer_1) / total_items if total_items > 0 else 0

    def compute_meta_potential(self) -> float:
        video_cache_index = self.env.mec_cache.policy.video_idx
        potential = 0.0

        if self.video_hist_long.maxlen > 0:
            for vid in video_cache_index:
                if vid != -1:
                    prob = self.video_freq_long.get(vid, 0) / self.video_hist_long.maxlen
                    potential += prob

        return potential

    def compute_ctrl_potential(self) -> float:
        video_cache_index = self.env.mec_cache.policy.video_idx
        tile_cache_index = self.env.mec_cache.policy.tile_idx
        potential = 0.0
        
        if self.tile_hist_long.maxlen > 0:
            for vid_idx, vid in enumerate(video_cache_index):
                if vid == -1: continue
                
                for tile in tile_cache_index[vid_idx]:
                    if tile != -1:
                        prob = self.tile_freq_long.get((vid, tile), 0) / self.tile_hist_long.maxlen
                        potential += prob

        return potential

    def _update_window(self, hist_queue: deque, freq_dict: Dict, item):
        if len(hist_queue) == hist_queue.maxlen:
            old_item = hist_queue.popleft()
            freq_dict[old_item] -= 1
            if freq_dict[old_item] == 0:
                del freq_dict[old_item]
        hist_queue.append(item)
        freq_dict[item] += 1


class NetworkAdapter:
    def __init__(self, cfg: Any, env: Any, feature_adapter: Any):
        self.env = env
        self.cfg = cfg
        self.features = feature_adapter

        self.C = self.cfg.cache_size  # paper's cache capacity (videos)
        self.k = self.cfg.viewport    # paper's tiles per video (enhancement)

        print(f"NetworkAdapter initialized with capacity: {self.C} videos, {self.k} tiles per video")

    def build_observation(self, req) -> np.ndarray:

        if req is None:
            return np.zeros(self.cfg.state_dim, dtype=np.float32) 

        vid = req["video"]
        viewport = req["viewport"]

        cache = self.env.mec_cache.policy.cache

        x_s = np.zeros(len(cache), dtype=np.float32)
        x_l = np.zeros(len(cache), dtype=np.float32)

        for i, (video, tile) in enumerate(cache):
            if video == -1:
                continue

            if tile == -1:
                x_s[i] = self.features.video_freq_short.get(video, 0) / self.features.video_hist_short.maxlen
                x_l[i] = self.features.video_freq_long.get(video, 0) / self.features.video_hist_long.maxlen
            else:
                x_s[i] = self.features.tile_freq_short.get((video, tile), 0) / self.features.tile_hist_short.maxlen
                x_l[i] = self.features.tile_freq_long.get((video, tile), 0) / self.features.tile_hist_long.maxlen

        y_s = np.array(
            [self.features.video_freq_short.get(vid, 0) / self.features.video_hist_short.maxlen], dtype=np.float32
        )
        y_l = np.array(
            [self.features.video_freq_long.get(vid, 0) / self.features.video_hist_long.maxlen], dtype=np.float32
        )

        z_s = np.zeros(len(viewport), dtype=np.float32)
        z_l = np.zeros(len(viewport), dtype=np.float32)

        for i, tile in enumerate(viewport):
            z_s[i] = self.features.tile_freq_short.get((vid, tile), 0) / self.features.tile_hist_short.maxlen
            z_l[i] = self.features.tile_freq_long.get((vid, tile), 0) / self.features.tile_hist_long.maxlen
        
        features = np.concatenate([x_s, x_l, y_s, y_l, z_s, z_l], axis=0)
        return features

    def reset(self):
        obs, info = self.env.reset()
        self.features.reset_history()

        return obs, info
    
    def env_is_done(self) -> bool:
        return self.env.users_env.all_users_done()
    
