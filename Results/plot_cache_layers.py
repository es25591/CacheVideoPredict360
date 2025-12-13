import os
import csv
import argparse
from typing import List

import matplotlib.pyplot as plt


def read_results(csv_path: str):
    episodes: List[int] = []
    enhanced_hits: List[float] = []
    enhanced_misses: List[float] = []
    base_hits: List[float] = []
    base_misses: List[float] = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = [
            "episode",
            "enhanced_layer_cache_hits",
            "enhanced_layer_cache_misses",
            "base_layer_cache_hits",
            "base_layer_cache_misses",
        ]
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. Found: {reader.fieldnames}"
            )

        for row in reader:
            try:
                episodes.append(int(row["episode"]))
                enhanced_hits.append(float(row["enhanced_layer_cache_hits"]))
                enhanced_misses.append(float(row["enhanced_layer_cache_misses"]))
                base_hits.append(float(row["base_layer_cache_hits"]))
                base_misses.append(float(row["base_layer_cache_misses"]))
            except (ValueError, KeyError):
                # skip malformed rows
                continue

    return episodes, enhanced_hits, enhanced_misses, base_hits, base_misses


def plot_cache_layers(
    csv_path: str,
    out_path: str | None = None,
    title: str | None = None,
):
    episodes, enhanced_hits, enhanced_misses, base_hits, base_misses = read_results(csv_path)

    plt.figure(figsize=(10, 6))

    plt.plot(episodes, enhanced_hits, label="Enhanced Layer Hits", color="#1f77b4")
    plt.plot(episodes, enhanced_misses, label="Enhanced Layer Misses", color="#ff7f0e")
    plt.plot(episodes, base_hits, label="Base Layer Hits", color="#2ca02c")
    plt.plot(episodes, base_misses, label="Base Layer Misses", color="#d62728")

    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if title is None:
        title = os.path.basename(csv_path)
    plt.title(f"Cache Layer Hits/Misses — {title}")

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved figure to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cache layer hits/misses over episodes")
    parser.add_argument(
        "csv",
        help="Path to training results CSV with layer cache columns",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Optional path to save the figure (PNG). If omitted, shows interactively.",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Optional plot title override",
    )
    args = parser.parse_args()

    plot_cache_layers(args.csv, out_path=args.out, title=args.title)
