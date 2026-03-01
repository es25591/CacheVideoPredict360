import csv
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from collections.abc import Mapping


class AgentDebugger:
    def __init__(self):
        self.data = defaultdict(list)

    def _to_python(self, value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()          # scalar -> no []
            return value.tolist()            # vector/matrix -> list

        if isinstance(value, np.generic):
            return value.item()

        return value

    def log(self, key, value):
        self.data[key].append(self._to_python(value))

    def plot(self, key, title=None):
        values = np.asarray(self.data[key], dtype=float)

        plt.figure()
        plt.plot(values)
        plt.title(title or key)
        plt.xlabel("Training step")
        plt.show()

    def histogram(self, key, title=None):
        items = self.data.get(key, [])
        if not items:
            return

        values = np.ravel(np.asarray(items, dtype=float))
        if values.size == 0:
            return

        plt.figure()
        plt.hist(values, bins=30)
        plt.title(title or key)
        plt.show()

    def clear(self):
        self.data.clear()

    def plot_request_popularity_vs_cache_presence(
        self,
        feature_adapter,
        drl_policy,
        cache_history=None,
        top_k=None,
        title="Request Popularity vs Cache Presence",
    ):
        """
        Validate DRL cache behavior by plotting request rank vs cache presence.

        X-axis: request rank (1 = most requested in video_freq_long)
        Y-axis: cache presence (binary at episode end, or residency ratio if history provided)

        Args:
            feature_adapter: object exposing `video_freq_long` mapping.
            drl_policy: object exposing `video_idx` list of cached videos.
            cache_history: optional list of cache snapshots per step/decision.
                           each snapshot can be iterable video ids or a mapping.
            top_k: optional number of top-ranked videos to plot.
            title: matplotlib title.

        Returns:
            dict with ranked videos, frequencies, cache_presence, and correlation.
        """
        freq_map = getattr(feature_adapter, "video_freq_long", None)
        if not isinstance(freq_map, Mapping):
            raise ValueError("feature_adapter must expose mapping-like video_freq_long")

        request_freq = {
            int(vid): float(freq)
            for vid, freq in freq_map.items()
            if int(vid) >= 0 and float(freq) > 0
        }
        if not request_freq:
            raise ValueError("video_freq_long is empty; no data to plot")

        current_cache = {int(v) for v in getattr(drl_policy, "video_idx", []) if int(v) >= 0}

        residency = None
        if cache_history:
            total_steps = len(cache_history)
            presence_counter = defaultdict(int)

            for snapshot in cache_history:
                if isinstance(snapshot, Mapping):
                    videos = {int(v) for v in snapshot.keys() if int(v) >= 0}
                else:
                    videos = {int(v) for v in snapshot if int(v) >= 0}
                for vid in videos:
                    presence_counter[vid] += 1

            if total_steps > 0:
                residency = {
                    vid: presence_counter.get(vid, 0) / total_steps
                    for vid in request_freq.keys()
                }

        ranked_videos = sorted(
            request_freq.keys(),
            key=lambda vid: (-request_freq[vid], vid),
        )
        if top_k is not None:
            ranked_videos = ranked_videos[: int(top_k)]

        ranks = np.arange(1, len(ranked_videos) + 1, dtype=int)
        frequencies = np.array([request_freq[vid] for vid in ranked_videos], dtype=float)

        if residency is None:
            cache_presence = np.array(
                [1.0 if vid in current_cache else 0.0 for vid in ranked_videos],
                dtype=float,
            )
            y_label = "Is in Cache (0/1)"
        else:
            cache_presence = np.array([residency[vid] for vid in ranked_videos], dtype=float)
            y_label = "Cache Residency Ratio"

        corr = np.nan
        if len(frequencies) >= 2 and np.std(cache_presence) > 0:
            corr = float(np.corrcoef(frequencies, cache_presence)[0, 1])

        plt.figure(figsize=(8, 4.5))
        plt.scatter(ranks, cache_presence, alpha=0.7, s=24)
        plt.title(title)
        plt.xlabel("Request Rank (Zipf Popularity)")
        plt.ylabel(y_label)
        plt.ylim(-0.05, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"Popularity/Cache correlation (Pearson): {corr:.4f}")

        return {
            "ranked_videos": ranked_videos,
            "request_freq": frequencies,
            "cache_presence": cache_presence,
            "correlation": corr,
        }

    def save_results(self, filepath="log_results", format='pickle'):
        """
        Store the collected simulation results to a file.

        Args:
            filepath: Path where to save the results
            format: 'pickle', 'json', or 'csv'
        """
        serializable_data = {
            key: [self._to_python(v) for v in values]
            for key, values in self.data.items()
        }

        with open(f"{filepath}.json", "w") as f:
            json.dump(serializable_data, f, indent=2)

        keys = list(serializable_data.keys())
        rows = zip(*serializable_data.values())

        with open(f"{filepath}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(rows)

        # print(f"Results saved to {filepath}")


debug = AgentDebugger()


def plot_request_popularity_vs_cache_presence(
    feature_adapter,
    drl_policy,
    cache_history=None,
    top_k=None,
    title="Request Popularity vs Cache Presence",
):
    return debug.plot_request_popularity_vs_cache_presence(
        feature_adapter=feature_adapter,
        drl_policy=drl_policy,
        cache_history=cache_history,
        top_k=top_k,
        title=title,
    )