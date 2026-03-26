import random

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

        return reward, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory.clear()

class RolloutBuffer:
    def __init__(self, capacity: int = 2000):
        self.memory = deque(maxlen=capacity)

    def push(self, log_prob, value, reward, next_state, done):
        self.memory.append((log_prob, value, reward, next_state, done))

    def get_all(self):
        return self.memory

    def sample(self, batch_size: int = None):
        if batch_size is None:
            return self.memory
        return random.sample(self.memory, batch_size)

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
            reward += (self.gamma ** i) * transition[4]

        # reward = reward / self.n_step # Normalize reward by n_step to prevent large values

        return reward, self.n_step_buffer[-1][5], self.n_step_buffer[-1][6] 
    
    def sample(self, batch_size: int = None):
        if batch_size is None:
            return self.memory
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()