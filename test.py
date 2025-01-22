# test.py
import gymnasium as gym
from polynomial_env import PolynomialEnv
from stable_baselines3 import PPO

def main():
    # Initialize the environment
    env = PolynomialEnv(max_steps=20)
    
    # Load the trained model
    model = PPO.load("ppo_polynomial_agent")
    print("Model loaded.")
    
    # Reset the environment
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Render the initial state
    env.render()
    
    while not done:
        # Predict the action using the trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Apply the action
        obs, reward, done, truncated, info = env.step(action)
        
        # Accumulate the reward
        total_reward += reward
        
        # Render the current state
        env.render()
    
    print(f"Final Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()

