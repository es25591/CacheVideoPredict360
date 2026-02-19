import csv
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


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

        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump(dict(self.data), f)

        print(f"Results saved to {filepath}")


debug = AgentDebugger()