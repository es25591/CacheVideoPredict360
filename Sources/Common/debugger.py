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