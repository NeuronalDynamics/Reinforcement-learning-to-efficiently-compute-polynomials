# train.py
import gymnasium as gym
from polynomial_circuit_env import PolynomialCircuitEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def main():
    # Define the polynomial input.
    # For example, for the polynomial x^2 + 2xy + y^2:
    # n = 2 (two variables), d = 2 (degree 2), coeffs = [1, 2, 1]
    poly_input = (2, 2, [1, 2, 1])
    
    # Create the environment with the polynomial input.
    env = PolynomialCircuitEnv(poly_input=poly_input, max_steps=20)
    
    # (Optional) Check if the environment follows Gym's API.
    check_env(env, warn=True)
    
    # Create the RL agent using PPO with a multi-layer perceptron policy.
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent for a given number of timesteps.
    print("Starting training...")
    model.learn(total_timesteps=20000)
    print("Training completed.")
    
    # Save the trained model.
    model.save("ppo_polynomial_circuit_agent")
    print("Model saved as 'ppo_polynomial_circuit_agent.zip'.")
    
    # Close the environment.
    env.close()

if __name__ == "__main__":
    main()

