# train.py
import gymnasium as gym
from polynomial_env import PolynomialEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def main():
    # Initialize the environment
    env = PolynomialEnv(max_steps=20)
    
    # Optional: Check if the environment follows Gym's API
    check_env(env, warn=True)
    
    # Initialize the RL agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=20000)
    print("Training completed.")
    
    # Save the trained model
    model.save("ppo_polynomial_agent")
    print("Model saved as 'ppo_polynomial_agent.zip'")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
