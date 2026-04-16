import random
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 2000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class NStepReplayBuffer:
    def __init__(self, capacity, n_step, gamma):
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, s, a, r, ns, done):
        self.n_step_buffer.append((s, a, r, ns, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute N-step discounted reward
        # G = r1 + gamma*r2 + ... + gamma^(n-1)*rn
        reward, next_state, done_ = self._get_n_step_info()
        state, action, _, _, _ = self.n_step_buffer[0]
        self.memory.append((state, action, reward, next_state, done_))

    def _get_n_step_info(self):
        reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * transition[2]
            # reward += transition[2]
            
        # reward = reward / self.n_step
        
        return reward, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
    
    def compute_gae(self, rewards, values, next_value, dones, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards from batch (shape: [batch_size])
            values: List of state values from critic (shape: [batch_size])
            next_value: Value of the next state (scalar)
            dones: List of done flags (shape: [batch_size])
            lam: GAE lambda parameter (0.95 is standard)
            
        Returns:
            advantages: GAE advantages (shape: [batch_size])
            returns: TD targets for critic (shape: [batch_size])
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        
        # Process in reverse order
        next_value_t = next_value
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            # TD residual (TD error): δ = r + γV(s') - V(s)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE: A(t) = δ(t) + (λγ)δ(t+1) + (λγ)²δ(t+2) + ...
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Returns = Advantages + Values (for critic target)
        returns = advantages + values
        
        return advantages, returns
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory.clear()
        self.n_step_buffer.clear()

class RolloutBuffer:
    def __init__(self, capacity: int = 2000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_all(self):
        return self.memory

    def sample(self, batch_size: int = None):
        if batch_size is None:
            return self.memory
        return random.sample(self.memory, batch_size)

    def compute_gae(self, rewards, values, next_value, dones, gamma, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) over an ordered rollout.
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value_t = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

class NStepRolloutBuffer:
    def __init__(self, capacity, n_step, gamma):
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, log_prob, value, reward, next_state, done):
        self.n_step_buffer.append((log_prob, value, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        
        reward, next_state, done_ = self._get_n_step_info()
        log_prob, value, _, _, _ = self.n_step_buffer[0]
        self.memory.append((log_prob, value, reward, next_state, done_))

    def _get_n_step_info(self):
        reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * transition[2]

        # reward = reward / self.n_step # Normalize reward by n_step to prevent large values

        return reward, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
    
    def sample(self, batch_size: int = None):
        if batch_size is None:
            return self.memory
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()
        self.n_step_buffer.clear()