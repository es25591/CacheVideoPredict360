import torch

import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
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
    def __init__(self, state_dim, action_dim, num_heads=4, hidden_dim=512):
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

    def forward(self, x, enh_mask=None):
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))


        # Compute Base Q-values: Shape (batch_size, action_dim_base)
        q_base = self.base_head(x) 
        
        # Compute Enh Q-values: Shape (batch_size, 4, action_dim_enh)
        q_enh = [head(x) for head in self.enh_heads]
        q_enh = torch.stack(q_enh, dim=1) 
        
        if enh_mask is not None:
            if enh_mask.dim() == 1:
                # 1. Change [32] -> [32, 1, 1]
                # 2. This allows it to automatically broadcast across 4 heads and 5 actions
                enh_mask = enh_mask.view(-1, 1, 1)

            q_enh = q_enh.masked_fill(~enh_mask, -1e9)

        return q_base, q_enh
    
class HierarchicalDQNet(nn.Module):
    def __init__(self, state_dim, action_dim_base, action_dim_enh, hidden_dim=256, hidden_dims=None):
        super().__init__()
        self.action_dim_enh = action_dim_enh

        if hidden_dims is None:
            hidden_dims = (hidden_dim,)
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        else:
            hidden_dims = tuple(hidden_dims)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        last_dim = hidden_dims[-1]
        
        # --- 1. State Encoding Shared Layers ---
        encoder_layers = []
        in_dim = state_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, dim))
            encoder_layers.append(nn.ReLU())
            in_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- 2. Base Action Head (Layer-1 Selection) ---
        self.base_head = nn.Linear(last_dim, action_dim_base)
        
        # --- 3. Conditional Enhancement Heads (Layer-2 Parameters) ---
        self.enh_heads = nn.ModuleList([
            nn.Linear(last_dim, action_dim_enh) for _ in range(4)
        ])

    def forward(self, state):
        # Step 1: Encode state and predict Base Q-values first
        latent = self.encoder(state)
        q_base = self.base_head(latent) # Shape: (batch, action_dim_base)
        
        # Step 2: Predict Q-values for all enhancement dimensions
        # These are now conditioned on the state features, awaiting base action selection
        q_enh = torch.stack([head(latent) for head in self.enh_heads], dim=1) 
        # Shape: (batch, 4, action_dim_enh)
        
        # Step 3: Top-Down Selection
        # Select base action first, then the parameters corresponding to that context
        chosen_base_action = q_base.argmax(dim=1)
        chosen_enh_actions = q_enh.argmax(dim=2)
        
        return q_base, q_enh, chosen_base_action, chosen_enh_actions
    
class ParameterizedDQN(nn.Module):
    def __init__(self, state_dim, action_dim_base, action_dim_enh, hidden_dim=256):
        super().__init__()
        self.action_dim_base = action_dim_base
        self.action_dim_enh = action_dim_enh

        # 1. State Encoding
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. Base Action Head
        self.base_head = nn.Linear(hidden_dim, action_dim_base)
        
        # 3. Enhancement Head (Vectorized for speed)
        # Input: latent + one_hot_base
        self.enh_shared = nn.Sequential(
            nn.Linear(hidden_dim + action_dim_base, hidden_dim),
            nn.ReLU()
        )
        # We produce all possible enhancement Q-values at once
        self.enh_final = nn.Linear(hidden_dim, action_dim_base * action_dim_enh)

    def forward(self, state, chosen_base_action=None):
        # Ensure state is (Batch, Dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        latent = self.encoder(state)
        q_base = self.base_head(latent)

        # Inference logic
        if chosen_base_action is None:
            chosen_base_action = q_base.argmax(dim=1)

        # Ensure chosen_base_action is (Batch,)
        base_one_hot = F.one_hot(chosen_base_action, num_classes=self.action_dim_base).float()

        # Step 4: Compute enhancement Q-values
        combined_input = torch.cat([latent, base_one_hot], dim=1)
        x_enh = self.enh_shared(combined_input)
        
        # Reshape output to (Batch, Base_Action_Idx, Enh_Action_Idx)
        q_enh_all = self.enh_final(x_enh).view(-1, self.action_dim_base, self.action_dim_enh)

        return q_base, q_enh_all, chosen_base_action
    

class DiscretePDQN(nn.Module):
    def __init__(self, state_dim, action_dim_base, action_dim_enh, hidden_dim=256):
        super().__init__()
        self.action_dim_enh = action_dim_enh

        # --- 1. Parameter Network (Enhancement Heads) ---
        self.enh_shared = nn.Linear(state_dim, hidden_dim)
        self.enh_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim_enh) for _ in range(4)
        ])

        # --- 2. Q-Network (Base Action) ---
        # The input dimension is the state PLUS the 4 one-hot encoded enhancement actions
        self.base_shared = nn.Linear(state_dim + (4 * action_dim_enh), hidden_dim)
        self.base_head = nn.Linear(hidden_dim, action_dim_base)

    def forward(self, state, chosen_enh_actions=None):
        # Step 1: Predict Q-values for the Enhancements based ONLY on the state
        x_enh = F.relu(self.enh_shared(state))
        q_enh = torch.stack([head(x_enh) for head in self.enh_heads], dim=1) # Shape: (batch, 4, action_dim_enh)

        # Step 2: Determine which enhancements to feed into the Base Network
        if chosen_enh_actions is None:
            # If acting or calculating next-state targets, greedily pick the best enhancements
            chosen_enh_actions = q_enh.argmax(dim=2) # Shape: (batch, 4)
            
        # Step 3: One-hot encode the chosen actions so the linear layers understand them
        enh_one_hot = F.one_hot(chosen_enh_actions, num_classes=self.action_dim_enh).float()
        enh_one_hot = enh_one_hot.view(state.size(0), -1) # Flatten to (batch, 4 * action_dim_enh)
        
        # Step 4: Predict the Base Q-values CONDITIONED on those chosen enhancements
        combined_input = torch.cat([state, enh_one_hot], dim=1)
        x_base = F.relu(self.base_shared(combined_input))
        q_base = self.base_head(x_base) # Shape: (batch, action_dim_base)
        
        return q_base, q_enh, chosen_enh_actions

# With dependency between Base and Enhancements (i.e. Enhancement heads get the Base action as input)
class DependentFocusQNetwork(nn.Module):
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
        
        self.base_emb = nn.Embedding(action_dim_base, 16) 
        
        # Enhancement Focus Heads (4 heads)
        self.enh_heads = nn.ModuleList([
            nn.Linear(h3 + 16, action_dim_enh) for _ in range(4)
        ])

    def forward(self, x, base_action=None, enh_mask=None):
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Compute Base Q-values: Shape (batch_size, action_dim_base)
        q_base = self.base_head(x) 

        if base_action is None:
            return q_base, None

        z_base = self.base_emb(base_action) # (batch, 16)

        x_enh = torch.cat([x, z_base], dim=-1)

        q_enh = [head(x_enh) for head in self.enh_heads]
        q_enh = torch.stack(q_enh, dim=1)
        
        if enh_mask is not None:
            q_enh = q_enh.masked_fill(~enh_mask, -1e9)
            
        return q_base, q_enh

class A2CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=None, hidden_dims=(2048, 1024, 512)):
        super(A2CNetwork, self).__init__()        

        if hidden_dims is None:
            hidden_dims = (hidden_dim,)
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        else:
            hidden_dims = tuple(hidden_dims)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        last_dim = hidden_dims[-1]
        
        # --- 1. State Encoding Shared Layers ---
        encoder_layers = []
        in_dim = state_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, dim))
            encoder_layers.append(nn.ReLU())
            in_dim = dim
        self.common = nn.Sequential(*encoder_layers)

        self.action_dim = action_dim

        # Actor head: Outputs probabilities for each tile/action
        self.actor = nn.Sequential(
            nn.Linear(last_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head: Outputs a single scalar value for the state
        self.critic = nn.Linear(last_dim, 1)

    def forward(self, x, actions=None):
        x = self.common(x)

        value = self.critic(x)
        probs = self.actor(x)
        
        dist = torch.distributions.Categorical(probs=probs)

        if actions is not None:
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return value, log_probs, entropy

        return value, dist.probs
        
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from collections import deque
# import random
# from typing import Tuple, Dict, List, Optional

# class ConditionalE_PQN_DQN(nn.Module):
#     """
#     E-PQN with Conditional Action Selection using DQN architecture
#     Q(s, a, x) with action selection following π(a, x | s) = π_base(a | s) * π_enh(x | s, a)
#     """

#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         param_dim: int,  # Dimension of enhancement parameter x
#         hidden_dim: int = 256,
#         param_continuous: bool = True,  # True for continuous x, False for discrete
#         param_bins: int = 10  # Number of bins if discretizing continuous params
#     ):
#         super().__init__()

#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.param_dim = param_dim
#         self.param_continuous = param_continuous

#         # For continuous parameters, we'll discretize them for DQN
#         # or use a hybrid approach (DQN for actions, deterministic for params)
#         if param_continuous:
#             # Discretize each continuous parameter dimension into bins
#             self.param_bins = param_bins
#             self.discrete_param_dim = param_bins ** param_dim
#             self.output_dim = action_dim * self.discrete_param_dim
#         else:
#             self.discrete_param_dim = param_dim
#             self.output_dim = action_dim * param_dim

#         # Shared feature extractor
#         self.feature_net = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )

#         # Q-value output for all (action, enhancement) combinations
#         self.q_net = nn.Linear(hidden_dim, self.output_dim)
        
#         # For tracking which actions are available (if needed)
#         self.action_mask = None
        
#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass computing Q-values for all (action, enhancement) pairs
        
#         Args:
#             state: State tensor [batch_size, state_dim]
            
#         Returns:
#             Q-values tensor [batch_size, action_dim * discrete_param_dim]
#         """
#         features = self.feature_net(state)
#         q_values = self.q_net(features)
#         return q_values
    
#     def get_q_tensor(self, state: torch.Tensor) -> torch.Tensor:
#         """
#         Reshape Q-values into [batch_size, action_dim, discrete_param_dim]
#         for easier indexing
#         """
#         q_values = self.forward(state)
#         return q_values.view(-1, self.action_dim, self.discrete_param_dim)
    
#     def get_action_enhancement_q(
#         self, 
#         state: torch.Tensor, 
#         action: torch.Tensor, 
#         param_idx: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Get Q-value for specific (action, enhancement) pair
        
#         Args:
#             state: State tensor [batch_size, state_dim]
#             action: Action indices [batch_size]
#             param_idx: Enhancement parameter indices [batch_size]
            
#         Returns:
#             Q-values [batch_size]
#         """
#         q_tensor = self.get_q_tensor(state)  # [batch, action_dim, param_dim]
        
#         # Gather specific Q-values
#         batch_size = state.shape[0]
#         q_values = q_tensor[torch.arange(batch_size), action, param_idx]
        
#         return q_values
    
#     def discretize_enhancement(self, continuous_x: torch.Tensor) -> torch.Tensor:
#         """
#         Convert continuous enhancement parameters to discrete indices
#         """
#         if not self.param_continuous:
#             return continuous_x
        
#         # Normalize to [0, 1] if needed (assuming input is already scaled appropriately)
#         # For simplicity, assume continuous_x is in [-1, 1] or [0, 1] range
        
#         # Map to bin indices
#         bins = torch.linspace(-1, 1, self.param_bins).to(continuous_x.device)
        
#         if self.param_dim == 1:
#             # Single parameter dimension
#             indices = torch.bucketize(continuous_x, bins) - 1
#             indices = torch.clamp(indices, 0, self.param_bins - 1)
#         else:
#             # Multiple parameter dimensions
#             indices = []
#             for i in range(self.param_dim):
#                 dim_indices = torch.bucketize(continuous_x[..., i], bins) - 1
#                 dim_indices = torch.clamp(dim_indices, 0, self.param_bins - 1)
#                 indices.append(dim_indices)
            
#             # Convert multi-dim indices to single index
#             indices = torch.stack(indices, dim=-1)
#             # Convert to flat index
#             flat_indices = indices[..., 0] * (self.param_bins ** 1)
#             if self.param_dim > 1:
#                 for i in range(1, self.param_dim):
#                     flat_indices += indices[..., i] * (self.param_bins ** (i+1))
#             indices = flat_indices
        
#         return indices
    
#     def continuous_from_discrete(self, param_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Convert discrete enhancement indices back to continuous values
#         """
#         if not self.param_continuous:
#             return param_idx
        
#         if self.param_dim == 1:
#             # Single dimension
#             bin_centers = torch.linspace(-1, 1, self.param_bins).to(param_idx.device)
#             return bin_centers[param_idx]
#         else:
#             # Multi-dimensional - convert flat index to multi-index
#             batch_size = param_idx.shape[0]
#             continuous_x = torch.zeros(batch_size, self.param_dim).to(param_idx.device)
            
#             for i in range(self.param_dim):
#                 # Extract indices for each dimension
#                 if i == 0:
#                     dim_indices = param_idx // (self.param_bins ** (self.param_dim - 1 - i))
#                 else:
#                     # This is simplified - proper multi-index conversion needed
#                     dim_indices = (param_idx // (self.param_bins ** (self.param_dim - 1 - i))) % self.param_bins
                
#                 bin_centers = torch.linspace(-1, 1, self.param_bins).to(param_idx.device)
#                 continuous_x[:, i] = bin_centers[dim_indices]
            
#             return continuous_x


# class ReplayBuffer:
#     """Experience replay buffer for DQN"""
    
#     def __init__(self, capacity: int = 100000):
#         self.buffer = deque(maxlen=capacity)
    
#     def push(
#         self,
#         state: np.ndarray,
#         action: int,
#         param_idx: int,
#         reward: float,
#         next_state: np.ndarray,
#         done: bool
#     ):
#         self.buffer.append((state, action, param_idx, reward, next_state, done))
    
#     def sample(self, batch_size: int) -> Tuple:
#         batch = random.sample(self.buffer, batch_size)
#         state, action, param_idx, reward, next_state, done = map(np.stack, zip(*batch))
#         return (
#             torch.FloatTensor(state),
#             torch.LongTensor(action),
#             torch.LongTensor(param_idx),
#             torch.FloatTensor(reward),
#             torch.FloatTensor(next_state),
#             torch.FloatTensor(done)
#         )
    
#     def __len__(self):
#         return len(self.buffer)


# class E_PQN_DQNAgent:
#     """
#     DQN Agent for E-PQN with conditional action selection
#     Implements action selection based on π(a, x | s) derived from Q-values
#     """
    
#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         param_dim: int,
#         learning_rate: float = 1e-4,
#         gamma: float = 0.99,
#         epsilon: float = 1.0,
#         epsilon_min: float = 0.01,
#         epsilon_decay: float = 0.995,
#         buffer_capacity: int = 100000,
#         batch_size: int = 64,
#         target_update: int = 100,
#         param_continuous: bool = True,
#         param_bins: int = 10,
#         device: str = "cpu"
#     ):
#         self.action_dim = action_dim
#         self.param_dim = param_dim
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.batch_size = batch_size
#         self.target_update = target_update
#         self.device = device
#         self.update_counter = 0
        
#         # Discretization info
#         self.param_continuous = param_continuous
#         self.param_bins = param_bins
#         self.discrete_param_dim = param_bins ** param_dim if param_continuous else param_dim
        
#         # Q-networks
#         self.q_network = ConditionalE_PQN_DQN(
#             state_dim=state_dim,
#             action_dim=action_dim,
#             param_dim=param_dim,
#             param_continuous=param_continuous,
#             param_bins=param_bins
#         ).to(device)
        
#         self.target_network = ConditionalE_PQN_DQN(
#             state_dim=state_dim,
#             action_dim=action_dim,
#             param_dim=param_dim,
#             param_continuous=param_continuous,
#             param_bins=param_bins
#         ).to(device)
        
#         self.target_network.load_state_dict(self.q_network.state_dict())
#         self.target_network.eval()
        
#         self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
#         self.memory = ReplayBuffer(buffer_capacity)
        
#     def select_action(
#         self,
#         state: np.ndarray,
#         eval_mode: bool = False
#     ) -> Tuple[int, int, Optional[np.ndarray]]:
#         """
#         Select action and enhancement using ε-greedy policy derived from Q-values
        
#         This implements: π(a, x | s) derived from Q(s, a, x)
#         In ε-greedy form: random with prob ε, greedy with prob 1-ε
        
#         Returns:
#             action: Selected action index
#             param_idx: Selected enhancement parameter index
#             continuous_x: Continuous enhancement values (if param_continuous=True)
#         """
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
#         # ε-greedy exploration
#         if not eval_mode and np.random.random() < self.epsilon:
#             # Random exploration
#             action = np.random.randint(self.action_dim)
#             param_idx = np.random.randint(self.discrete_param_dim)
#         else:
#             # Greedy selection based on Q-values
#             with torch.no_grad():
#                 q_tensor = self.q_network.get_q_tensor(state_tensor)  # [1, action_dim, param_dim]
                
#                 # Find best (action, param) pair
#                 q_values_flat = q_tensor.view(1, -1)  # [1, action_dim * param_dim]
#                 best_idx = torch.argmax(q_values_flat, dim=1).item()
                
#                 # Convert flat index to (action, param) indices
#                 action = best_idx // self.discrete_param_dim
#                 param_idx = best_idx % self.discrete_param_dim
        
#         # Convert to continuous if needed
#         continuous_x = None
#         if self.param_continuous:
#             param_tensor = torch.LongTensor([param_idx]).to(self.device)
#             continuous_x = self.q_network.continuous_from_discrete(param_tensor)
#             continuous_x = continuous_x.cpu().numpy().squeeze()
        
#         return action, param_idx, continuous_x
    
#     def compute_enhancement_probabilities(
#         self,
#         state: torch.Tensor,
#         action: torch.Tensor,
#         temperature: float = 1.0
#     ) -> torch.Tensor:
#         """
#         Compute π_enh(x | s, a) using softmax over Q-values for fixed action
        
#         This implements the conditional enhancement policy derived from Q-values:
#         π_enh(x | s, a) = exp(Q(s, a, x) / τ) / Σ_x' exp(Q(s, a, x') / τ)
#         """
#         q_tensor = self.q_network.get_q_tensor(state)  # [batch, action_dim, param_dim]
        
#         # Get Q-values for the specific action
#         batch_size = state.shape[0]
#         action_q = q_tensor[torch.arange(batch_size), action, :]  # [batch, param_dim]
        
#         # Apply softmax to get probabilities
#         probs = F.softmax(action_q / temperature, dim=-1)
        
#         return probs
    
#     def compute_base_action_probabilities(
#         self,
#         state: torch.Tensor,
#         temperature: float = 1.0
#     ) -> torch.Tensor:
#         """
#         Compute π_base(a | s) by marginalizing over enhancement parameters:
#         π_base(a | s) = Σ_x π_enh(x | s, a) * [usually from Q-values]
        
#         Here we derive it from Q-values using softmax over max or mean Q
#         """
#         q_tensor = self.q_network.get_q_tensor(state)  # [batch, action_dim, param_dim]
        
#         # Use max over parameters as Q-value for action
#         max_q_per_action, _ = torch.max(q_tensor, dim=-1)  # [batch, action_dim]
        
#         # Softmax over actions
#         probs = F.softmax(max_q_per_action / temperature, dim=-1)
        
#         return probs
    
#     def store_transition(
#         self,
#         state: np.ndarray,
#         action: int,
#         param_idx: int,
#         reward: float,
#         next_state: np.ndarray,
#         done: bool
#     ):
#         """Store experience in replay buffer"""
#         self.memory.push(state, action, param_idx, reward, next_state, done)
    
#     def update(self) -> Dict[str, float]:
#         """Update Q-network using experience replay"""
#         if len(self.memory) < self.batch_size:
#             return {'loss': 0.0, 'q_mean': 0.0}
        
#         # Sample batch
#         states, actions, param_idxs, rewards, next_states, dones = self.memory.sample(self.batch_size)
#         states = states.to(self.device)
#         actions = actions.to(self.device)
#         param_idxs = param_idxs.to(self.device)
#         rewards = rewards.to(self.device)
#         next_states = next_states.to(self.device)
#         dones = dones.to(self.device)
        
#         # Current Q-values
#         current_q = self.q_network.get_action_enhancement_q(states, actions, param_idxs)
        
#         # Target Q-values
#         with torch.no_grad():
#             # Get Q-values for next state
#             next_q_tensor = self.target_network.get_q_tensor(next_states)  # [batch, action_dim, param_dim]
            
#             # Double DQN: use online network to select action, target network to evaluate
#             next_q_tensor_online = self.q_network.get_q_tensor(next_states)
            
#             # Find best (action, param) for each next state using online network
#             next_q_flat = next_q_tensor_online.view(self.batch_size, -1)
#             best_indices = torch.argmax(next_q_flat, dim=1)
            
#             best_actions = best_indices // self.discrete_param_dim
#             best_params = best_indices % self.discrete_param_dim
            
#             # Evaluate using target network
#             next_q = self.target_network.get_action_enhancement_q(
#                 next_states, best_actions, best_params
#             )
            
#             # Compute target
#             target_q = rewards + self.gamma * (1 - dones) * next_q
        
#         # Compute loss (Huber loss for stability)
#         loss = F.smooth_l1_loss(current_q, target_q)
        
#         # Optimize
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
#         self.optimizer.step()
        
#         # Update target network
#         self.update_counter += 1
#         if self.update_counter % self.target_update == 0:
#             self.target_network.load_state_dict(self.q_network.state_dict())
        
#         # Decay epsilon
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
#         return {
#             'loss': loss.item(),
#             'q_mean': current_q.mean().item(),
#             'epsilon': self.epsilon
#         }
    
#     def save(self, path: str):
#         """Save model"""
#         torch.save({
#             'q_network': self.q_network.state_dict(),
#             'target_network': self.target_network.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'epsilon': self.epsilon
#         }, path)
    
#     def load(self, path: str):
#         """Load model"""
#         checkpoint = torch.load(path)
#         self.q_network.load_state_dict(checkpoint['q_network'])
#         self.target_network.load_state_dict(checkpoint['target_network'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         self.epsilon = checkpoint['epsilon']


# # Example environment wrapper for E-PQN
# class E_PQNEnvironment:
#     """
#     Example environment wrapper for testing the E-PQN DQN agent
#     """
    
#     def __init__(self, action_dim: int = 4, param_dim: int = 2, param_continuous: bool = True):
#         self.action_dim = action_dim
#         self.param_dim = param_dim
#         self.param_continuous = param_continuous
#         self.state_dim = 10  # Example state dimension
        
#     def reset(self) -> np.ndarray:
#         """Reset environment and return initial state"""
#         return np.random.randn(self.state_dim)
    
#     def step(self, action: int, param: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
#         """
#         Execute action with enhancement parameter
        
#         Args:
#             action: Action index
#             param: Enhancement parameter (continuous or discrete)
            
#         Returns:
#             next_state, reward, done, info
#         """
#         # Example dynamics (replace with your actual environment)
#         next_state = np.random.randn(self.state_dim)
        
#         # Example reward function that depends on both action and param
#         if self.param_continuous:
#             # Reward depends on how well param matches some target
#             target_param = np.array([0.5, -0.3])  # Example target
#             param_error = np.linalg.norm(param - target_param)
#             reward = -param_error + np.random.randn() * 0.1
#         else:
#             reward = np.random.randn()
        
#         done = np.random.random() < 0.05  # 5% chance of episode ending
        
#         return next_state, reward, done, {}


# # Training loop example
# def train_e_pqn_dqn():
#     """Example training loop for E-PQN DQN agent"""
    
#     # Create environment and agent
#     env = E_PQNEnvironment(action_dim=4, param_dim=2, param_continuous=True)
#     agent = E_PQN_DQNAgent(
#         state_dim=env.state_dim,
#         action_dim=env.action_dim,
#         param_dim=env.param_dim,
#         learning_rate=1e-4,
#         gamma=0.99,
#         epsilon=1.0,
#         epsilon_min=0.01,
#         epsilon_decay=0.995,
#         buffer_capacity=100000,
#         batch_size=64,
#         target_update=100,
#         param_continuous=True,
#         param_bins=10
#     )
    
#     num_episodes = 1000
#     max_steps = 200
#     update_frequency = 4
    
#     episode_rewards = []
    
#     for episode in range(num_episodes):
#         state = env.reset()
#         episode_reward = 0
        
#         for step in range(max_steps):
#             # Select action and enhancement
#             action, param_idx, continuous_x = agent.select_action(state)
            
#             # Execute in environment
#             next_state, reward, done, info = env.step(action, continuous_x)
            
#             # Store transition
#             agent.store_transition(state, action, param_idx, reward, next_state, done)
            
#             # Update agent
#             if step % update_frequency == 0:
#                 loss_info = agent.update()
            
#             state = next_state
#             episode_reward += reward
            
#             if done:
#                 break
        
#         episode_rewards.append(episode_reward)
        
#         # Logging
#         if episode % 100 == 0:
#             avg_reward = np.mean(episode_rewards[-100:])
#             print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
#     return agent, episode_rewards


# if __name__ == "__main__":
#     # Quick test
#     agent, rewards = train_e_pqn_dqn()
    
#     # Test in evaluation mode
#     env = E_PQNEnvironment()
#     state = env.reset()
#     action, param_idx, continuous_x = agent.select_action(state, eval_mode=True)
#     print(f"Evaluation - Action: {action}, Param idx: {param_idx}")
#     if continuous_x is not None:
#         print(f"Continuous param: {continuous_x}")