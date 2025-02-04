# train_evaluate_plot.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from polynomial_circuit_env import PolynomialCircuitEnv

def train_agent(poly_input, max_steps, total_timesteps=100000):
    """
    Create the environment, check it, and train the PPO agent for a given
    number of timesteps.
    
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
    
    # Create the RL agent using PPO with a multi-layer perceptron policy.
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent.
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")
    
    # Save the model.
    model.save("ppo_polynomial_circuit_agent_hard")
    print("Model saved as 'ppo_polynomial_circuit_agent_hard.zip'.")
    
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
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        rewards_per_episode.append(total_reward)
        # We consider the episode a success if the final circuit cost is 0.
        # The observation is defined as: [num_add, num_mul, num_scalar, cost, m]
        # So the cost is the 4th element.
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
    # Make the task harder by choosing a more challenging polynomial.
    # For example, consider a homogeneous polynomial of degree 3 in 3 variables.
    # Suppose our polynomial has coefficients (in lexicographical order) for the monomials.
    # For demonstration, we use a dummy coefficient list (you may use more meaningful ones).
    n = 3        # number of variables
    d = 3        # degree
    # For a homogeneous polynomial in 3 variables of degree 3,
    # there are C(n+d-1, d) = C(3+3-1, 3) = C(5, 3) = 10 monomials.
    coeffs = [np.random.randint(1, 5) for _ in range(10)]
    poly_input = (n, d, coeffs)
    
    # Set a maximum number of steps per episode.
    max_steps = 40
    
    # Train the agent with a longer training period.
    model = train_agent(poly_input, max_steps, total_timesteps=100000)
    
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

