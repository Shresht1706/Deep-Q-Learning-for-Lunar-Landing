import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Lander_Env import Agent, env, state_size, number_actions  # Import agent and env setup

# === Architectures to Test ===
architectures = {
    'Tiny': [32, 32],
    'Base': [64, 64],
    'Wide': [128, 128],
    'Deep': [256, 128, 64],
    'Tanh': [64, 64],
}

activations = {
    'Tiny': nn.ReLU,
    'Base': nn.ReLU,
    'Wide': nn.ReLU,
    'Deep': nn.ReLU,
    'Tanh': nn.Tanh,
}

# === Dynamic Network Injection ===
class NeuNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation):
        super(NeuNet, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# === Train Runner ===
def run_training(arch_name, layers, activation_fn, runs=5, episodes=1000):
    all_rewards = []
    all_times = []
    success_eps = []

    for run in range(runs):
        print(f"\n[Run {run + 1}/5] - {arch_name}")
        agent = Agent(state_size, number_actions)
        agent.local_qnetwork = NeuNet(state_size, number_actions, layers, activation_fn).to(agent.device)
        agent.target_qnetwork = NeuNet(state_size, number_actions, layers, activation_fn).to(agent.device)

        episode_rewards = []
        start_time = time.time()
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0

            while not done:
                action = agent.act(obs)
                next_obs, reward, done, _, _ = env.step(action)
                agent.step(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_reward += reward

            episode_rewards.append(ep_reward)

        end_time = time.time()
        all_rewards.append(episode_rewards)
        all_times.append(end_time - start_time)

        # Success detection
        success_ep = -1
        for i in range(100, len(episode_rewards)):
            if np.mean(episode_rewards[i-100:i]) > 200:
                success_ep = i
                break
        success_eps.append(success_ep)

    return all_rewards, all_times, success_eps

# === Main Runner ===
if __name__ == "__main__":
    results = {}

    for arch in architectures:
        rewards, times, success = run_training(
            arch_name=arch,
            layers=architectures[arch],
            activation_fn=activations[arch]
        )
        results[arch] = {
            'rewards': rewards,
            'times': times,
            'success_eps': success
        }

    print("\n===== SUMMARY =====")
    for arch, data in results.items():
        avg_time = np.mean(data['times'])
        valid_successes = [s for s in data['success_eps'] if s != -1]
        avg_success = np.mean(valid_successes) if valid_successes else "Never"
        print(f"{arch}: Avg Time = {avg_time:.2f}s | Avg Success Ep = {avg_success}")

    # Plot Reward Curves
    plt.figure(figsize=(12, 6))
    for arch in results:
        mean_rewards = np.mean(np.array(results[arch]['rewards']), axis=0)
        plt.plot(mean_rewards, label=arch)
    plt.title("Average Reward per Episode (All Architectures)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("architecture_rewards_comparison.png")
    plt.show()
