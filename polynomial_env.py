# polynomial_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PolynomialEnv(gym.Env):
    """
    A 1-player environment where the agent tries to reduce the cost
    of computing a polynomial by applying rewrites or merges.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps=20):
        super(PolynomialEnv, self).__init__()
        self.max_steps = max_steps
        
        # Action space:
        # 0 = merge-add, 1 = merge-mul, 2 = scalar factor, 3 = no-op
        self.action_space = spaces.Discrete(4)
        
        # Observation space:
        # [#add gates, #mul gates, #scalar gates, total_cost]
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        
        self.reset()

    def _compute_cost(self, circuit_rep):
        """
        Example cost: sum of #add gates, #mul gates, and #scalar gates
        """
        return circuit_rep['add'] + circuit_rep['mul'] + circuit_rep['scalar']

    def _make_observation(self, circuit_rep):
        """
        Convert circuit_rep into a 4-dim vector: [#add, #mul, #scalar, total_cost]
        """
        cost = self._compute_cost(circuit_rep)
        return np.array([
            circuit_rep['add'],
            circuit_rep['mul'],
            circuit_rep['scalar'],
            cost
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize circuit with random number of gates
        self.state = {
            'add': np.random.randint(1, 5),
            'mul': np.random.randint(1, 5),
            'scalar': np.random.randint(0, 3)
        }
        self.steps = 0
        obs = self._make_observation(self.state)
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1
        
        # Save previous cost
        prev_cost = self._compute_cost(self.state)
        
        # Define how each action transforms the circuit.
        if action == 0:
            # Attempt to combine two add gates or something that reduces #add
            if self.state['add'] > 0:
                self.state['add'] = max(0, self.state['add'] - 1)
        elif action == 1:
            # Attempt to combine two mul gates
            if self.state['mul'] > 0:
                self.state['mul'] = max(0, self.state['mul'] - 1)
        elif action == 2:
            # Attempt to factor out a scalar
            if self.state['scalar'] > 0:
                self.state['scalar'] = max(0, self.state['scalar'] - 1)
        elif action == 3:
            # No operation
            pass
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate new cost
        cost = self._compute_cost(self.state)
        
        # Reward is the reduction in cost
        reward = prev_cost - cost
        
        # If no reduction, assign a small negative reward
        if reward <= 0:
            reward = -1.0
        
        # Check if done
        done = (self.steps >= self.max_steps) or (cost == 0)
        truncated = False

        obs = self._make_observation(self.state)
        info = {}
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.steps}")
        print(f"Gates - Add: {self.state['add']}, Mul: {self.state['mul']}, Scalar: {self.state['scalar']}")
        print(f"Total Cost: {self._compute_cost(self.state)}\n")

    def close(self):
        pass
