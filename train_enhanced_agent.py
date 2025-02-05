# train_enhanced_agent.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from enhanced_circuit_env import EnhancedCircuitEnv

def train_agent(total_timesteps=200000, max_steps=100, cost_threshold=5):
    """
    Create the EnhancedCircuitEnv, check it, and train a PPO agent.
    Curriculum learning is enabled.
    """
    env = EnhancedCircuitEnv(cost_threshold=cost_threshold, max_steps=max_steps,
                             curriculum=True, curriculum_max=1000)
    check_env(env, warn=True)
    
    # Define a custom deep network with two hidden layers (256 neurons each) for both policy and value.
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                policy_kwargs=policy_kwargs)
    
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")
    model.save("ppo_enhanced_circuit_agent")
    print("Model saved as 'ppo_enhanced_circuit_agent.zip'.")
    env.close()
    return model

def inspect_policy_value(model, env):
    """
    Sample a state from the environment and inspect the outputs from the policy and value heads.
    
    This implementation extracts the latent representations using the current StableBaselines3 API.
    """
    obs, _ = env.reset()
    # Convert observation to a torch tensor using the model's helper.
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    
    # Set the policy to evaluation mode.
    model.policy.eval()
    
    with torch.no_grad():
        # Extract features from the observation.
        features = model.policy.extract_features(obs_tensor)
        # Obtain latent representations for policy and value using the MLP extractor.
        latent_pi, latent_vf = model.policy.mlp_extractor(features)
        # Get the action distribution from the latent policy representation.
        action_dist = model.policy._get_action_dist_from_latent(latent_pi)
        # Get the value estimate from the latent value representation.
        value = model.policy.value_net(latent_vf)
    
    # Extract probabilities from the action distribution.
    probs = action_dist.distribution.probs.cpu().numpy()[0]
    value_est = value.cpu().numpy()[0]
    
    print("Sample state observation:", obs)
    print("Policy head probabilities:", probs)
    print("Value head estimate:", value_est)

def evaluate_model(model, max_steps=100, num_episodes=100, cost_threshold=5):
    """
    Evaluate the trained model over a number of episodes.
    """
    env = EnhancedCircuitEnv(cost_threshold=cost_threshold, max_steps=max_steps,
                             curriculum=True, curriculum_max=1000)
    rewards = []
    success_count = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        
        if info.get('cost', np.inf) <= cost_threshold:
            success_count += 1
    
    env.close()
    success_rate = success_count / num_episodes * 100
    return rewards, success_rate

def plot_results(rewards, success_rate):
    episodes = np.arange(1, len(rewards) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Enhanced Circuit Proof Generation: Cumulative Reward per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Success rate: {success_rate:.1f}% of episodes achieved cost <= threshold.")

def main():
    total_timesteps = 200000
    max_steps = 100
    cost_threshold = 5
    
    # Train the agent.
    model = train_agent(total_timesteps=total_timesteps, max_steps=max_steps, cost_threshold=cost_threshold)
    
    # Create an instance of the environment to inspect policy and value outputs.
    env = EnhancedCircuitEnv(cost_threshold=cost_threshold, max_steps=max_steps,
                             curriculum=True, curriculum_max=1000)
    print("\nInspecting policy and value head outputs on a sample state:")
    inspect_policy_value(model, env)
    
    # Evaluate the trained agent.
    num_eval_episodes = 100
    rewards, success_rate = evaluate_model(model, max_steps=max_steps, num_episodes=num_eval_episodes, cost_threshold=cost_threshold)
    
    avg_reward = np.mean(rewards)
    print(f"Average cumulative reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    plot_results(rewards, success_rate)

if __name__ == "__main__":
    main()

