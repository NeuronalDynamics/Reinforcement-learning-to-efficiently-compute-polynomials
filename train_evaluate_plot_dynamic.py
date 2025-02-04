# train_evaluate_plot_dynamic.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from math import comb  # Python 3.8+ for binomial coefficient
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from polynomial_circuit_env import PolynomialCircuitEnv

def train_agent(poly_input, max_steps, total_timesteps=200000):
    """
    Create the environment, check it, and train a PPO agent with a custom policy
    architecture for a given number of timesteps.
    
    Parameters:
      poly_input: tuple (n, d, coeffs) representing the polynomial.
      max_steps: maximum steps per episode.
      total_timesteps: total timesteps to train the agent.
    
    Returns:
      model: the trained PPO model.
    """
    # Create the environment with the given polynomial input.
    env = PolynomialCircuitEnv(poly_input=poly_input, max_steps=max_steps)
    
    # (Optional) Check if the environment conforms to Gym's API.
    check_env(env, warn=True)
    
    # Define a custom policy network architecture.
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )
    
    # Create the RL agent using PPO with the custom network architecture.
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                policy_kwargs=policy_kwargs)
    
    # Train the agent.
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")
    
    # Save the model.
    model.save("ppo_polynomial_circuit_agent_dynamic")
    print("Model saved as 'ppo_polynomial_circuit_agent_dynamic.zip'.")
    
    # Close the environment.
    env.close()
    
    return model

def evaluate_model(model, poly_input, max_steps, num_episodes=50):
    """
    Evaluate the trained model on the environment for a number of episodes.
    
    Parameters:
      model: the trained RL model.
      poly_input: tuple (n, d, coeffs) representing the polynomial.
      max_steps: maximum steps per episode.
      num_episodes: number of episodes for evaluation.
      
    Returns:
      rewards_per_episode: list of cumulative rewards for each episode.
      success_rate: fraction of episodes that reached an ideal circuit (cost==0).
    """
    env = PolynomialCircuitEnv(poly_input=poly_input, max_steps=max_steps)
    rewards_per_episode = []
    success_count = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        
        rewards_per_episode.append(total_reward)
        # The observation is [num_add, num_mul, num_scalar, cost, m]
        # If cost (4th element) is 0, we consider the circuit fully simplified.
        if obs[3] == 0:
            success_count += 1

    env.close()
    success_rate = success_count / num_episodes * 100
    return rewards_per_episode, success_rate

def plot_results(rewards, success_rate):
    """
    Plot the cumulative rewards per episode and print the success rate.
    
    Parameters:
      rewards: list of rewards for each evaluation episode.
      success_rate: percentage of episodes that reached cost 0.
    """
    episodes = np.arange(1, len(rewards)+1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='b')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Evaluation: Cumulative Rewards per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Success rate: {success_rate:.1f}% of episodes reached an ideal circuit (cost = 0).")

def main():
    # Make the task more challenging by choosing dynamic n and d.
    # You can adjust these values to change the complexity.
    n = 4        # number of variables (for example)
    d = 4        # degree of the homogeneous polynomial
    
    # Calculate the number of monomials for a homogeneous polynomial:
    # m = C(n+d-1, d)
    m = comb(n + d - 1, d)
    print(f"Generating polynomial with n = {n}, d = {d}, m = {m} monomials.")
    
    # Generate random coefficients for all monomials.
    coeffs = [np.random.randint(1, 5) for _ in range(m)]
    poly_input = (n, d, coeffs)
    
    # Set a maximum number of steps per episode.
    max_steps = 100  # increased steps for a harder task
    
    # Train the agent with a longer training period for a harder problem.
    model = train_agent(poly_input, max_steps, total_timesteps=200000)
    
    # Evaluate the trained model.
    num_eval_episodes = 50
    rewards, success_rate = evaluate_model(model, poly_input, max_steps, num_episodes=num_eval_episodes)
    
    # Print average reward.
    avg_reward = np.mean(rewards)
    print(f"Average cumulative reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    
    # Plot the results.
    plot_results(rewards, success_rate)

if __name__ == "__main__":
    main()

