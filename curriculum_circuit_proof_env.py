# curriculum_circuit_proof_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CurriculumCircuitProofEnv(gym.Env):
    """
    A Gymnasium environment for a simplified version of proof generation via arithmetic circuit
    simplification. The goal is to transform an arithmetic circuit computing a polynomial 
    f(x_1, …, x_n) into a more efficient one by applying rewriting moves.
    
    The circuit is represented by a state vector with enhanced features:
      [num_add, num_mul, num_scalar, depth, total_cost, total_gates]
    
    where:
      - num_add: number of addition gates,
      - num_mul: number of multiplication gates,
      - num_scalar: number of scalar multiplications,
      - depth: an estimated depth of the circuit,
      - total_cost: a weighted sum cost,
      - total_gates: total number of operations.
      
    The cost is computed as:
      cost = w1 * num_add + w2 * num_mul + w3 * num_scalar + w4 * depth,
    with fixed weights.
    
    Actions (4 discrete moves):
      0: Combine adjacent addition gates (reduce num_add and depth),
      1: Combine adjacent multiplication gates (reduce num_mul and depth),
      2: Factor out a common scalar (reduce num_scalar, small reduction in depth),
      3: No operation.
      
    The episode terminates when the cost is below a threshold or when a maximum number
    of steps is reached.
    
    Additionally, this environment supports curriculum learning. In curriculum mode,
    the initial circuit starts simple (easy) and gradually increases in complexity
    over episodes.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, cost_threshold=5, max_steps=100, curriculum=True, curriculum_max=1000):
        super().__init__()
        self.max_steps = max_steps
        self.cost_threshold = cost_threshold
        self.curriculum = curriculum
        self.curriculum_max = curriculum_max  # number of episodes over which difficulty increases
        
        # Cost weights (tunable)
        self.w1, self.w2, self.w3, self.w4 = 1.0, 1.0, 0.5, 0.5
        
        # Enhanced state representation: 6 dimensions
        # [num_add, num_mul, num_scalar, depth, total_cost, total_gates]
        self.observation_space = spaces.Box(low=0, high=500, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        
        # For curriculum learning: track episode count
        self.episode_count = 0
        
        self.reset()
    
    def _compute_cost(self, state):
        return (self.w1 * state['num_add'] +
                self.w2 * state['num_mul'] +
                self.w3 * state['num_scalar'] +
                self.w4 * state['depth'])
    
    def _make_observation(self, state):
        total_cost = self._compute_cost(state)
        total_gates = state['num_add'] + state['num_mul'] + state['num_scalar']
        return np.array([
            state['num_add'],
            state['num_mul'],
            state['num_scalar'],
            state['depth'],
            total_cost,
            total_gates
        ], dtype=np.float32)
    
    def _initial_state(self):
        """
        Generate an initial circuit state.
        If curriculum is enabled, the circuit starts simple and gradually increases in complexity.
        Easy (low complexity): [5, 5, 2, 5]
        Hard (high complexity): [20, 20, 10, 10]
        """
        if self.curriculum:
            factor = min(1.0, self.episode_count / self.curriculum_max)
            # Linear interpolation between easy and hard values
            num_add = int(round(5 + factor * (20 - 5)))
            num_mul = int(round(5 + factor * (20 - 5)))
            num_scalar = int(round(2 + factor * (10 - 2)))
            depth = int(round(5 + factor * (10 - 5)))
        else:
            num_add, num_mul, num_scalar, depth = 20, 20, 10, 10
        
        state = {
            'num_add': num_add,
            'num_mul': num_mul,
            'num_scalar': num_scalar,
            'depth': depth
        }
        return state
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Increment episode count (for curriculum learning)
        self.episode_count += 1
        
        self.steps = 0
        self.state = self._initial_state()
        obs = self._make_observation(self.state)
        info = {}
        return obs, info
    
    def step(self, action):
        self.steps += 1
        prev_cost = self._compute_cost(self.state)
        
        # Apply rewriting moves
        if action == 0:
            # Combine adjacent additions: reduce num_add by 1 (if > 1) and depth by 1 (if possible)
            if self.state['num_add'] > 1:
                self.state['num_add'] -= 1
                if self.state['depth'] > 1:
                    self.state['depth'] -= 1
        elif action == 1:
            # Combine adjacent multiplications: reduce num_mul and depth similarly
            if self.state['num_mul'] > 1:
                self.state['num_mul'] -= 1
                if self.state['depth'] > 1:
                    self.state['depth'] -= 1
        elif action == 2:
            # Factor out a common scalar: reduce num_scalar and slightly reduce depth
            if self.state['num_scalar'] > 0:
                self.state['num_scalar'] -= 1
                if self.state['depth'] > 1:
                    self.state['depth'] -= 0.5
        elif action == 3:
            # No operation
            pass
        else:
            raise ValueError(f"Invalid action: {action}")
        
        new_cost = self._compute_cost(self.state)
        reward = prev_cost - new_cost  # positive if cost is reduced
        if reward <= 0:
            reward = -1.0  # penalize non-improving moves
        
        done = (new_cost <= self.cost_threshold) or (self.steps >= self.max_steps)
        truncated = False
        obs = self._make_observation(self.state)
        info = {'cost': new_cost}
        
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        cost = self._compute_cost(self.state)
        total_gates = self.state['num_add'] + self.state['num_mul'] + self.state['num_scalar']
        print(f"Step: {self.steps}")
        print(f"State: num_add={self.state['num_add']}, num_mul={self.state['num_mul']}, "
              f"num_scalar={self.state['num_scalar']}, depth={self.state['depth']}")
        print(f"Total cost: {cost}, Total gates: {total_gates}")
    
    def close(self):
        pass

