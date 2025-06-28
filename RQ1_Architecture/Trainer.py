import time
import torch
import torch.nn as nn
import numpy as np
import csv
from collections import deque
from Lander_Env import Agent, env, state_size, number_actions

learning_rate = 5e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not detected. Running on CPU.")

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

def run_training(arch_name, layers, activation_fn, episodes=1000, max_timesteps=1000):
    print(f"\nStarting training for architecture: {arch_name} on device: {device}")
    agent = Agent(state_size, number_actions)
    agent.device = device
    agent.local_qnetwork = NeuNet(state_size, number_actions, layers, activation_fn).to(device)
    agent.target_qnetwork = NeuNet(state_size, number_actions, layers, activation_fn).to(device)
    agent.optimizer = torch.optim.Adam(agent.local_qnetwork.parameters(), lr=learning_rate)

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    all_scores = []

    start_time = time.time()
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        score = 0
        for t in range(max_timesteps):
            action = agent.act(state.cpu().numpy(), epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            agent.step(state.cpu().numpy(), action, reward, next_state, done)
            state = next_state_tensor
            score += reward
            if done:
                break
        all_scores.append(score)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % 100 == 0:
            print(f"Episode {episode} | Time Elapsed: {time.time() - start_time:.2f}s | Score = {score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    final_avg_score = np.mean(all_scores[949:1000]) if len(all_scores) >= 1000 else np.mean(all_scores[-50:])#last 50 scores to make it slightly more fair

    print(f"Final average score (episodes 900-1000): {final_avg_score:.2f}")

    torch.save(agent.local_qnetwork.state_dict(), f"checkpoint_{arch_name}.pth")

    with open("RQ1_results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([arch_name, elapsed_time, final_avg_score])

if __name__ == "__main__":
    architectures = {
        'Tiny': [32, 32],
        #'Base': [64, 64],
        #'Wide': [128, 128],
        #'Deep': [256, 128, 64],
    }

    for arch in architectures:
        for run in range(5):
            print(f"\n=== Run {run+3}/5 for architecture: {arch} ===")
            run_training(
                arch_name=f"{arch}_run{run+1}",
                layers=architectures[arch],
                activation_fn= nn.ReLU
            )
