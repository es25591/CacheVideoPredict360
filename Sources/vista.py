import numpy as np
import gymnasium as gym

from gymnasium import spaces

# vista.py
# A simple OpenAI Gym environment skeleton.
# Usage: import gym; import vista; env = vista.VistaEnv(); obs = env.reset(); obs, rew, done, info = env.step(env.action_space.sample())


class VistaEnv(gym.Env):
    """
    VistaEnv: minimal continuous 2D navigation environment.
    State: [x, y, vx, vy]
    Action: continuous acceleration in x,y (shape=(2,))
    Reward: negative L2 distance to a goal position (higher is better when closer).
    Episode ends when agent reaches goal (within tol) or after max_steps.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, max_steps: int = 200, goal_radius: float = 0.1):
        super().__init__()
        self.max_steps = int(max_steps)
        self.goal_radius = float(goal_radius)

        # Action: accelerations in x and y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: x, y, vx, vy with reasonable bounds
        obs_high = np.array([100.0, 100.0, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # internal state
        self.state = np.zeros(4, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.rng = np.random.default_rng()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [None if seed is None else int(seed)]

    def reset(self, seed=None):
        """
        Reset environment.
        Returns:
            obs (np.ndarray): initial observation
        """
        if seed is not None:
            self.seed(seed)
        # randomize start near origin, random goal somewhere in [-5,5]^2
        self.state[:2] = self.rng.uniform(low=-1.0, high=1.0, size=2)
        self.state[2:] = 0.0
        self.goal = self.rng.uniform(low=-5.0, high=5.0, size=2)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        """
        Apply action (acceleration). Simple Euler integration:
            v <- v + a * dt
            x <- x + v * dt
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        dt = 0.1
        # update velocity and position
        self.state[2:] = self.state[2:] + action * dt
        self.state[:2] = self.state[:2] + self.state[2:] * dt

        self.step_count += 1

        # reward = negative distance to goal
        dist = np.linalg.norm(self.state[:2] - self.goal)
        reward = -dist

        done = bool(dist <= self.goal_radius or self.step_count >= self.max_steps)

        info = {"is_success": bool(dist <= self.goal_radius), "goal": self.goal.copy(), "dist": float(dist)}

        return self._get_obs(), float(reward), done, info

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pos = self.state[:2]
        if mode == "rgb_array":
            # produce a simple 64x64 image with agent and goal dots
            size = 64
            img = np.zeros((size, size, 3), dtype=np.uint8)
            # mapping from world coords [-6,6] to image coords [0,size)
            def to_px(p):
                clip = np.clip((p + 6.0) / 12.0, 0.0, 1.0)
                return (clip * (size - 1)).astype(int)
            gx, gy = to_px(self.goal)
            ax, ay = to_px(pos)
            img[gy-1:gy+2, gx-1:gx+2] = [0, 255, 0]  # goal green
            img[ay-1:ay+2, ax-1:ax+2] = [255, 0, 0]  # agent red
            return img
        else:
            print(f"Step {self.step_count}: pos={pos}, goal={self.goal}")
            return None

    def close(self):
        return

if __name__ == "__main__":
    # tiny demo: random policy for a few episodes
    env = VistaEnv()
    for ep in range(3):
        obs = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            total += r
        print(f"Episode {ep} total_reward={total:.2f} success={info.get('is_success')}")
    env.close()