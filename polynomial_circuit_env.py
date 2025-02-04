# polynomial_circuit_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PolynomialCircuitEnv(gym.Env):
    """
    Environment where the goal is to generate an efficient arithmetic circuit
    that computes a given polynomial. The polynomial is provided as a tuple
    (n, d, coeffs) where:
      - n: number of variables,
      - d: degree of the polynomial,
      - coeffs: list of coefficients arranged in lexicographical order.
    
    The candidate circuit is abstracted as counters for operations:
      - num_add: number of addition gates,
      - num_mul: number of multiplication gates,
      - num_scalar: number of scalar multiplications.
    
    The overall cost is defined as: cost = num_add + num_mul + num_scalar.
    
    Actions:
      0: Combine addition gates (reduce num_add if > 0)
      1: Combine multiplication gates (reduce num_mul if > 0)
      2: Factor out a scalar (simulate improvement; reduce num_scalar if > 0)
      3: No operation
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, poly_input=(2, 2, [1, 2, 1]), max_steps=20):
        """
        Initialize the environment.
        
        Parameters:
          poly_input: tuple (n, d, coeffs) representing the polynomial.
          max_steps: maximum number of steps per episode.
        """
        super().__init__()
        
        # Save the polynomial parameters
        self.n, self.d, self.coeffs = poly_input
        self.m = len(self.coeffs)  # number of monomials
        
        self.max_steps = max_steps
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Observation space:
        # We encode the candidate circuit state and include the number of monomials (m).
        # Observation vector: [num_add, num_mul, num_scalar, cost, m]
        # cost is redundant but helps the agent understand progress.
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        
        self.reset()
    
    def _initial_circuit(self):
        """
        Generate an initial candidate circuit from the polynomial.
        For simplicity, we assume:
          - num_add = m - 1  (to sum the m monomials)
          - num_mul = m      (each monomial is computed by one multiplication)
          - num_scalar = 0   (initially, no scalar factors)
        """
        state = {
            'num_add': max(self.m - 1, 0),
            'num_mul': self.m,
            'num_scalar': 0
        }
        return state
    
    def _compute_cost(self, circuit_state):
        """
        Compute the cost of the circuit as the sum of its operations.
        """
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
            self.m  # include the number of monomials (from the input polynomial)
        ], dtype=np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        # Initialize the candidate circuit using the input polynomial.
        self.state = self._initial_circuit()
        self.steps = 0
        obs = self._make_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        """
        Execute the given action and update the state.
        
        Returns:
          obs: the new observation,
          reward: the improvement in cost (or a penalty),
          done: whether the episode has ended,
          truncated: always False in this simple environment,
          info: an (empty) dictionary.
        """
        self.steps += 1
        
        # Record the previous cost
        prev_cost = self._compute_cost(self.state)
        
        # Apply the chosen action
        if action == 0:
            # Combine addition gates if possible
            if self.state['num_add'] > 0:
                self.state['num_add'] = max(0, self.state['num_add'] - 1)
        elif action == 1:
            # Combine multiplication gates if possible
            if self.state['num_mul'] > 0:
                self.state['num_mul'] = max(0, self.state['num_mul'] - 1)
        elif action == 2:
            # Factor out a scalar if possible.
            # For our simulation, if there is any potential for improvement,
            # we simulate by adding a bonus reduction. If no scalar operations exist,
            # we simulate a possibility by creating one and then reducing it.
            if self.state['num_scalar'] > 0:
                self.state['num_scalar'] = max(0, self.state['num_scalar'] - 1)
            else:
                # For demonstration, allow a one-time creation and reduction of scalar gate.
                self.state['num_scalar'] = 0  # This action is effective only once.
        elif action == 3:
            # No operation; state remains unchanged.
            pass
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Compute the new cost after the action.
        new_cost = self._compute_cost(self.state)
        
        # Reward is the reduction in cost.
        reward = prev_cost - new_cost
        
        # If there is no improvement, penalize the agent.
        if reward <= 0:
            reward = -1.0
        
        # Determine if the episode is finished.
        # In our case, we can end if we reach a cost of 0 (ideal circuit) or exceed max_steps.
        done = (self.steps >= self.max_steps) or (new_cost == 0)
        truncated = False
        
        # Generate the observation.
        obs = self._make_observation()
        info = {}
        
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Print the current state of the environment.
        """
        cost = self._compute_cost(self.state)
        print(f"Step: {self.steps}")
        print(f"Current Circuit State: Additions: {self.state['num_add']}, "
              f"Multiplications: {self.state['num_mul']}, Scalars: {self.state['num_scalar']}")
        print(f"Total Cost: {cost}")
        print(f"Polynomial Monomials (m): {self.m}\n")
    
    def close(self):
        """
        Cleanup (if any) when the environment is closed.
        """
        pass

