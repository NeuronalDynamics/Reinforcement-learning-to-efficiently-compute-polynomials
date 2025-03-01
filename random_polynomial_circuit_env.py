# random_polynomial_circuit_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import comb  # Python 3.8+ for binomial coefficient
import random

class RandomPolynomialCircuitEnv(gym.Env):
    """
    Environment where the goal is to generate an efficient arithmetic circuit
    that computes a given polynomial. In each episode, the polynomial is randomly
    generated by sampling:
      - n: number of variables (from n_range),
      - d: degree of the polynomial (from d_range),
      - coeffs: list of coefficients (from coeff_range)
    
    The circuit is abstracted as counters for operations:
      - num_add: number of addition gates,
      - num_mul: number of multiplication gates,
      - num_scalar: number of scalar multiplications.
    
    The overall cost is defined as: cost = num_add + num_mul + num_scalar.
    
    Actions:
      0: Combine addition gates (reduce num_add if > 0)
      1: Combine multiplication gates (reduce num_mul if > 0)
      2: Factor out a scalar (reduce num_scalar if > 0; if 0, simulate one-time reduction)
      3: No operation
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, n_range=(2, 5), d_range=(2, 5), coeff_range=(1, 5), max_steps=100):
        """
        Parameters:
          n_range: tuple specifying the range (min, max) for the number of variables.
          d_range: tuple specifying the range (min, max) for the degree.
          coeff_range: tuple specifying the range (min, max) for the coefficients.
          max_steps: maximum number of steps per episode.
        """
        super().__init__()
        
        self.n_range = n_range
        self.d_range = d_range
        self.coeff_range = coeff_range
        self.max_steps = max_steps

        # Observation vector: [num_add, num_mul, num_scalar, cost, m]
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        # 4 discrete actions as defined above.
        self.action_space = spaces.Discrete(4)
        
        self.reset()
    
    def _sample_polynomial(self):
        # Sample n and d uniformly from their ranges.
        self.n = random.randint(*self.n_range)
        self.d = random.randint(*self.d_range)
        # Compute number of monomials for a homogeneous polynomial: m = C(n+d-1, d)
        self.m = comb(self.n + self.d - 1, self.d)
        # Generate random coefficients (each in [coeff_range[0], coeff_range[1]-1]).
        self.coeffs = [random.randint(self.coeff_range[0], self.coeff_range[1]-1) for _ in range(self.m)]
        # (Optional) Debug print:
        # print(f"Sampled polynomial: n={self.n}, d={self.d}, m={self.m}, coeffs={self.coeffs}")
    
    def _initial_circuit(self):
        """
        Generate an initial candidate circuit based on the polynomial.
        For simplicity, assume:
          - num_add = m - 1  (to sum the m monomials)
          - num_mul = m      (each monomial is computed by one multiplication)
          - num_scalar = 0   (initially, no scalar operations)
        """
        state = {
            'num_add': max(self.m - 1, 0),
            'num_mul': self.m,
            'num_scalar': 0
        }
        return state
    
    def _compute_cost(self, circuit_state):
        """Compute the cost of the circuit as the sum of its operations."""
        return circuit_state['num_add'] + circuit_state['num_mul'] + circuit_state['num_scalar']
    
    def _make_observation(self):
        """
        Create the observation vector from the current circuit state.
        Observation: [num_add, num_mul, num_scalar, cost, m]
        """
        cost = self._compute_cost(self.state)
        obs = np.array([
            self.state['num_add'],
            self.state['num_mul'],
            self.state['num_scalar'],
            cost,
            self.m  # the number of monomials (proxy for complexity)
        ], dtype=np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset the environment by sampling a new polynomial and initializing the circuit."""
        super().reset(seed=seed)
        self._sample_polynomial()
        self.state = self._initial_circuit()
        self.steps = 0
        obs = self._make_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        """Apply the given action, update the circuit, and return observation, reward, done, etc."""
        self.steps += 1
        prev_cost = self._compute_cost(self.state)
        
        # Apply action
        if action == 0:
            # Combine addition gates if possible
            if self.state['num_add'] > 0:
                self.state['num_add'] = max(0, self.state['num_add'] - 1)
        elif action == 1:
            # Combine multiplication gates if possible
            if self.state['num_mul'] > 0:
                self.state['num_mul'] = max(0, self.state['num_mul'] - 1)
        elif action == 2:
            # Factor out a scalar: if exists, reduce it; if none, simulate one-time benefit.
            if self.state['num_scalar'] > 0:
                self.state['num_scalar'] = max(0, self.state['num_scalar'] - 1)
            else:
                self.state['num_scalar'] = 0
        elif action == 3:
            # No operation
            pass
        else:
            raise ValueError(f"Invalid action: {action}")
        
        new_cost = self._compute_cost(self.state)
        reward = prev_cost - new_cost
        if reward <= 0:
            reward = -1.0
        
        done = (self.steps >= self.max_steps) or (new_cost == 0)
        truncated = False
        obs = self._make_observation()
        info = {}
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render the current circuit state and polynomial complexity."""
        cost = self._compute_cost(self.state)
        print(f"Step: {self.steps}")
        print(f"Current Circuit State: Additions: {self.state['num_add']}, "
              f"Multiplications: {self.state['num_mul']}, Scalars: {self.state['num_scalar']}")
        print(f"Total Cost: {cost}, m (monomials): {self.m}")
    
    def close(self):
        pass

