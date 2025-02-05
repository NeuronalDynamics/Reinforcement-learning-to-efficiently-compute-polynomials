# circuit_proof_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CircuitProofEnv(gym.Env):
    """
    A Gymnasium environment for a simplified version of proof generation.
    The goal is to transform an arithmetic circuit that computes a polynomial
    f(x_1, …, x_n) into a more efficient one using a series of rewriting moves.
    
    The circuit is abstractly represented by a state vector:
      [num_add, num_mul, num_scalar, depth]
      
    where:
      - num_add: number of addition gates,
      - num_mul: number of multiplication gates,
      - num_scalar: number of scalar multiplications,
      - depth: an estimated depth of the circuit.
      
    The cost of the circuit is computed as:
      cost = w1 * num_add + w2 * num_mul + w3 * num_scalar + w4 * depth,
    where the weights (w1, w2, w3, w4) are hyperparameters.
    
    Actions:
      0: Combine adjacent addition gates (reduce num_add and possibly depth).
      1: Combine adjacent multiplication gates (reduce num_mul and possibly depth).
      2: Factor out a common scalar (reduce num_scalar and possibly reduce depth).
      3: No operation.
      
    The episode ends when the circuit is sufficiently efficient (cost <= threshold)
    or after a maximum number of steps.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, cost_threshold=5, max_steps=100):
        super().__init__()
        self.max_steps = max_steps
        self.cost_threshold = cost_threshold
        
        # Define the weights for cost (tune these as needed)
        self.w1, self.w2, self.w3, self.w4 = 1.0, 1.0, 0.5, 0.5
        
        # State: [num_add, num_mul, num_scalar, depth]
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        
        # 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        self.reset()
    
    def _initial_state(self):
        """
        Initialize a candidate circuit.
        For simplicity, we start with a circuit that is not efficient.
        """
        # Example: Start with 20 additions, 20 multiplications, 10 scalars, depth 10.
        state = {
            'num_add': 20,
            'num_mul': 20,
            'num_scalar': 10,
            'depth': 10
        }
        return state
    
    def _compute_cost(self, state):
        cost = (self.w1 * state['num_add'] +
                self.w2 * state['num_mul'] +
                self.w3 * state['num_scalar'] +
                self.w4 * state['depth'])
        return cost
    
    def _make_observation(self, state):
        return np.array([state['num_add'], state['num_mul'],
                         state['num_scalar'], state['depth']], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._initial_state()
        self.steps = 0
        obs = self._make_observation(self.state)
        info = {}
        return obs, info
    
    def step(self, action):
        self.steps += 1
        prev_cost = self._compute_cost(self.state)
        
        # Apply rewriting moves based on the action
        if action == 0:
            # Combine two additions: reduce num_add by 1 and depth by 1 (if possible)
            if self.state['num_add'] > 1:
                self.state['num_add'] -= 1
                if self.state['depth'] > 1:
                    self.state['depth'] -= 1
        elif action == 1:
            # Combine two multiplications: reduce num_mul by 1 and depth by 1 (if possible)
            if self.state['num_mul'] > 1:
                self.state['num_mul'] -= 1
                if self.state['depth'] > 1:
                    self.state['depth'] -= 1
        elif action == 2:
            # Factor out a common scalar: reduce num_scalar by 1 and possibly depth.
            if self.state['num_scalar'] > 0:
                self.state['num_scalar'] -= 1
                if self.state['depth'] > 1:
                    self.state['depth'] -= 0.5  # smaller effect on depth
        elif action == 3:
            # No operation.
            pass
        else:
            raise ValueError(f"Invalid action: {action}")
        
        new_cost = self._compute_cost(self.state)
        reward = prev_cost - new_cost  # positive reward if cost is reduced
        if reward <= 0:
            reward = -1.0  # penalize moves that do not improve
        
        done = (new_cost <= self.cost_threshold) or (self.steps >= self.max_steps)
        truncated = False
        
        obs = self._make_observation(self.state)
        info = {'cost': new_cost}
        
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        cost = self._compute_cost(self.state)
        print(f"Step: {self.steps}")
        print(f"State: {self.state}")
        print(f"Cost: {cost}")
    
    def close(self):
        pass

