import numpy as np
from ray import state
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
    
class MultiHeadQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, latent_dim=256):
        super(MultiHeadQNetwork, self).__init__()
        
        # 1. Shared Feature Extractor (The Backbone)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # 2. Main Task Head: Q-Values
        self.q_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 3. Aux Task Head: Forward Dynamics (Predict next state)
        # Takes the latent state + action to predict the next state
        self.dynamics_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

    def forward(self, state):
        latent = self.encoder(state)
        q_values = self.q_head(latent)
        return q_values, latent
        
    def predict_dynamics(self, latent, action_one_hot):
        # Concatenate latent representation with the action taken
        x = torch.cat([latent, action_one_hot], dim=-1)
        return self.dynamics_head(x)

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

class A2CSharedNetwork(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=None, 
        hidden_dims=(2048, 1024, 512)
    ):
        super(A2CSharedNetwork, self).__init__()        

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

class LFUHeuristic:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        
    def get_action(self, state):
        y_l = state[2*self.cache_size + 1]        
        x_l = y_l + state[self.cache_size:2*self.cache_size]

        return int(np.argmin(y_l + x_l))

    def get_heuristic_values(self, state):
        cache_size = self.cache_size

        y_s = state[2*cache_size]
        y_l = state[2*cache_size + 1]
        
        x_s = y_s + state[:cache_size]
        x_l = y_l + state[cache_size:2*cache_size]
        
        arg_max_l = np.argmin(x_l)
        
        return arg_max_l, x_l[arg_max_l]

class A2CNetwork(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=None, 
        hidden_dims=(2048, 1024, 512)
    ):
        super(A2CNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = (hidden_dim,)
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        else:
            hidden_dims = tuple(hidden_dims)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        # Actor head: Outputs probabilities for each tile/action
        actor_layers = []
        in_dim = state_dim
        for dim in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, dim))
            actor_layers.append(nn.ReLU())
            in_dim = dim
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic head: Outputs a single scalar value for the state
        critic_layers = []
        in_dim = state_dim
        for dim in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, dim))
            critic_layers.append(nn.ReLU())
            in_dim = dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        

    def forward(self, x, action=None):
        value = self.critic(x)
        logits = self.actor(x)

        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs=probs)

        if action is not None:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return value, log_prob, entropy

        return value, dist.probs


class A2CHeuNetwork(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=None, 
        hidden_dims=(2048, 1024, 512)
    ):
        super(A2CHeuNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = (hidden_dim,)
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        else:
            hidden_dims = tuple(hidden_dims)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        # Actor head: Outputs probabilities for each tile/action
        actor_layers = []
        in_dim = state_dim
        for dim in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, dim))
            actor_layers.append(nn.ReLU())
            in_dim = dim
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic head: Outputs a single scalar value for the state
        critic_layers = []
        in_dim = state_dim
        for dim in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, dim))
            critic_layers.append(nn.ReLU())
            in_dim = dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        value = self.critic(x)
        logits = self.actor(x)

        return value, logits


class PermutationInvariantEncoder(nn.Module):

    def __init__(self, slot_dim=1, embed_dim=64):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(slot_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

        self.rho = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: [batch, 251]

        x = x.unsqueeze(-1)      # [batch, 251, 1]

        slot_embed = self.phi(x) # [batch, 251, embed_dim]

        pooled = slot_embed.mean(dim=1)  # permutation invariant

        state_embed = self.rho(pooled)

        return state_embed