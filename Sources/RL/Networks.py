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
    

class FocusQNetwork(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim_base, 
        action_dim_enh, 
        hidden_dims=(256, 128, 64),
        dropout_p=0.2
    ):
        super().__init__()

        h1, h2, h3 = hidden_dims
        
        # Shared representation layers
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Base Focus Head (1 head)
        self.base_head = nn.Linear(h3, action_dim_base)
        
        # Enhancement Focus Heads (4 heads)
        self.enh_heads = nn.ModuleList([
            nn.Linear(h3, action_dim_enh) for _ in range(4)
        ])

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # Compute Base Q-values: Shape (batch_size, action_dim_base)
        q_base = self.base_head(x) 
        
        # Compute Enh Q-values: Shape (batch_size, 4, action_dim_enh)
        q_enh = [head(x) for head in self.enh_heads]
        q_enh = torch.stack(q_enh, dim=1) 
        
        return q_base, q_enh