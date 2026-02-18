import csv
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import defaultdict


class AgentDebugger:
    def __init__(self):
        self.data = defaultdict(list)

    def log(self, key, value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        self.data[key].append(np.atleast_1d(value))

    def plot(self, key, title=None):
        values = np.array(self.data[key])

        plt.figure()
        plt.plot(values)
        plt.title(title or key)
        plt.xlabel("Training step")
        plt.show()
        
    def histogram(self, key, title=None):
        items = self.data.get(key, [])
        if not items:
            return

        values = np.concatenate([np.ravel(np.asarray(v)) for v in items])
        if values.size == 0:
            return

        plt.figure()
        plt.hist(values, bins=30)
        plt.title(title or key)
        plt.show()
        
    def clear(self):
        self.data.clear()
        
    def save_results(self, filepath, format='pickle'):
        """
        Store the collected simulation results to a file.
        
        Args:
            filepath: Path where to save the results
            format: 'pickle', 'json', or 'csv'
        """
        # Ensure we have a dictionary of serializable lists for JSON/CSV
        serializable_data = {
            key: [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
            for key, values in self.data.items()
        }

        with open(f"{filepath}.json", 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
        # We assume all keys have the same number of logs (time steps)
        keys = list(serializable_data.keys())
        rows = zip(*serializable_data.values())
        
        with open(f"{filepath}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)  # Header
            writer.writerows(rows) # Data rows

        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(dict(self.data), f)
         
        print(f"Results saved to {filepath}")