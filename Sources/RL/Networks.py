import torch

import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        hidden = hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class MultiHeadQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=4, hidden_dim=128):
        super().__init__()

        # shared encoder
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # independent heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        z = self.shared(x)
        return torch.stack([h(z) for h in self.heads], dim=1)
    
