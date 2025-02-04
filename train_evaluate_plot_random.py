# train_evaluate_plot_random.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from random_polynomial_circuit_env import RandomPolynomialCircuitEnv

def train_agent(total_timesteps=200000, max_steps=100, 
                n_range=(2, 5), d_range=(2, 5), coeff_range=(1, 5)):
    """
    Create the environment with randomized polynomial parameters, check it, and train
    a PPO agent with a custom network architecture.
    
    Returns:
      model: the trained PPO model.
    """
    # Create environment instance with randomized polynomial parameters.
    env = RandomPolynomialCircuitEnv(n_range=n_range, d_range=d_range, 
                                       coeff_range=coeff_range, max_steps=max_steps)
    
    # (Optional) Verify environment compatibility.
    check_env(env, warn=True)
    
    # Define a custom policy network (two hidden layers of 256 neurons each for both policy and value function).
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                policy_kwargs=policy_kwargs)
    
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")
    
    model.save("ppo_polynomial_circuit_agent_random")
    print("Model saved as 'ppo_polynomial_circuit_agent_random.zip'.")
    env.close()
    return model

def evaluate_model(model, max_steps=100, num_episodes=50, 
                   n_range=(2, 5), d_range=(2, 5), coeff_range=(1, 5)):
    """
    Evaluate the trained model on the randomized environment over a number of episodes.
    
    Returns:
      rewards_per_episode: list of cumulative rewards for each episode.
      success_rate: percentage of episodes that reached a cost of 0.
    """
    env = RandomPolynomialCircuitEnv(n_range=n_range, d_range=d_range, 
                                       coeff_range=coeff_range, max_steps=max_steps)
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
        # Observation: [num_add, num_mul, num_scalar, cost, m]
        if obs[3] == 0:
            success_count += 1

    env.close()
    success_rate = success_count / num_episodes * 100
    return rewards_per_episode, success_rate

def plot_results(rewards, success_rate):
    """
    Plot cumulative rewards per episode and display the success rate.
    """
    episodes = np.arange(1, len(rewards) + 1)
    
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
    # Set ranges for a more general and challenging problem.
    n_range = (2, 6)      # number of variables can vary between 2 and 6
    d_range = (2, 6)      # degree can vary between 2 and 6
    coeff_range = (1, 10) # coefficients in [1, 9]
    
    max_steps = 1000
    total_timesteps = 200000  # Increase timesteps for more training
    
    # Train the agent.
    model = train_agent(total_timesteps=total_timesteps, max_steps=max_steps, 
                        n_range=n_range, d_range=d_range, coeff_range=coeff_range)
    
    # Evaluate the trained agent.
    num_eval_episodes = 500
    rewards, success_rate = evaluate_model(model, max_steps=max_steps, num_episodes=num_eval_episodes, 
                                             n_range=n_range, d_range=d_range, coeff_range=coeff_range)
    
    avg_reward = np.mean(rewards)
    print(f"Average cumulative reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    plot_results(rewards, success_rate)

if __name__ == "__main__":
    main()

