# test.py
import gymnasium as gym
from polynomial_circuit_env import PolynomialCircuitEnv
from stable_baselines3 import PPO
import numpy as np

def evaluate_model(model, env, num_episodes=10):
    """
    Run num_episodes in the environment using the given model,
    and compute the average cumulative reward.
    """
    all_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}:")

        while not done:
            # Use the trained model to predict the action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action in the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Optionally, render the current state
            env.render(mode='human')
            
            # End the episode if it's done or truncated
            if done or truncated:
                break

        print(f"Total reward for episode {episode + 1}: {total_reward}")
        all_rewards.append(total_reward)
    
    avg_reward = np.mean(all_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward}")
    return all_rewards

def main():
    # Define the polynomial input.
    # For example, for the polynomial x^2 + 2xy + y^2:
    # n = 2 (two variables), d = 2 (degree 2), coeffs = [1, 2, 1]
    poly_input = (2, 2, [1, 2, 1])
    
    # Create the environment using the same settings as during training.
    env = PolynomialCircuitEnv(poly_input=poly_input, max_steps=20)
    
    # Load the saved model.
    model = PPO.load("ppo_polynomial_circuit_agent")
    print("Model loaded. Starting evaluation...")
    
    # Evaluate the model.
    evaluate_model(model, env, num_episodes=10)
    
    # Optionally, close the environment.
    env.close()

if __name__ == "__main__":
    main()

