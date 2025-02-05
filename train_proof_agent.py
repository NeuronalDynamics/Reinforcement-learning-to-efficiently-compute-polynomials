# train_proof_agent.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from circuit_proof_env import CircuitProofEnv

def train_agent(total_timesteps=200000, max_steps=100):
    """
    Create the CircuitProofEnv environment, check it, and train a PPO agent.
    """
    env = CircuitProofEnv(max_steps=max_steps, cost_threshold=5)
    check_env(env, warn=True)
    
    # Use a custom policy network architecture (deeper network)
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                policy_kwargs=policy_kwargs)
    
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")
    model.save("ppo_circuit_proof_agent")
    print("Model saved as 'ppo_circuit_proof_agent.zip'.")
    env.close()
    return model

def evaluate_model(model, max_steps=100, num_episodes=100):
    """
    Evaluate the trained model over a number of episodes.
    """
    env = CircuitProofEnv(max_steps=max_steps, cost_threshold=5)
    rewards = []
    success_count = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        
        # Check if the proof was completed (i.e., cost is below threshold)
        if info.get('cost', np.inf) <= 5:
            success_count += 1
    
    env.close()
    success_rate = success_count / num_episodes * 100
    return rewards, success_rate

def plot_results(rewards, success_rate):
    episodes = np.arange(1, len(rewards)+1)
    
    plt.figure(figsize=(10,5))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Proof Generation: Cumulative Reward per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Success rate: {success_rate:.1f}% of episodes achieved cost <= threshold.")

def main():
    total_timesteps = 200000
    max_steps = 100  # adjust as needed (e.g., set higher for more complex proof search)
    
    # Train the agent
    model = train_agent(total_timesteps=total_timesteps, max_steps=max_steps)
    
    # Evaluate the trained agent
    num_eval_episodes = 100
    rewards, success_rate = evaluate_model(model, max_steps=max_steps, num_episodes=num_eval_episodes)
    
    avg_reward = np.mean(rewards)
    print(f"Average cumulative reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    plot_results(rewards, success_rate)

if __name__ == "__main__":
    main()

