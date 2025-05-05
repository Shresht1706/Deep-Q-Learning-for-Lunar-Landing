# Deep Q-Learning for Lunar Lander

This project implements a Deep Q-Network (DQN) agent to solve the Lunar Lander environment from the [Gymnasium](https://gymnasium.farama.org/) toolkit. It leverages PyTorch for building and training the model, and applies standard RL techniques such as experience replay, epsilon-greedy policy, and soft target updates.

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## Introduction

This notebook demonstrates how to use Deep Q-Learning to train an agent that can land a rocket in the Lunar Lander environment. The training process involves neural networks, memory replay, stochastic policies, and gradient-based learning.

---

## Installation

To get started, install the necessary dependencies using the commands below:

```bash
pip install gymnasium==1.0.0
pip install swig
pip install "gymnasium[box2d]"
pip install imageio
pip install ipython
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "imageio[ffmpeg]"
```

---

## Usage

1. Open the `Model.ipynb` notebook in Jupyter.
2. Run the cells sequentially to:
   - Install dependencies
   - Initialize the agent and environment
   - Train the model using DQN
   - Evaluate and optionally save the trained model

Training runs for up to 2000 episodes or until the average reward over 100 episodes reaches 200, whichever comes first.

---

## Features

- Deep Q-Network with feedforward architecture
- Experience replay for stable learning
- Epsilon-greedy policy for exploration/exploitation balance
- Soft target updates for improved convergence
- GPU acceleration via CUDA (if available)
- Video export of trained agent using `imageio`

---

## Dependencies

- Python 3.x
- Gymnasium (`gymnasium[box2d]`)
- PyTorch
- ImageIO (with ffmpeg)
- IPython
- SWIG

---

## Configuration

### Model Architecture

The neural network (`NeuNet`) consists of:

- Input Layer → Linear(64) → ReLU
- Hidden Layer → Linear(64) → ReLU
- Output Layer → Linear to action space

### Hyperparameters

- `learning_rate = 5e-4`
- `gamma = 0.99`
- `tau = 1e-3`
- `minibatch_size = 100`
- `replay_buffer_size = 1e5`
- `epsilon_start = 1.0`
- `epsilon_end = 0.01`
- `epsilon_decay = 0.995`
- `max_num_timesteps_per_ep = 1000`
- `number_ep = 2000`

---

## Documentation

### Key Components

- **NeuNet**: PyTorch neural network for Q-value prediction
- **Agent**:
  - `act(state, epsilon)`: Selects actions based on current policy
  - `step(...)`: Stores transitions and controls learning frequency
  - `learn(...)`: Updates the Q-network
  - `soft_update(...)`: Smoothly updates the target network
- **ReplayMemory**: Circular buffer for experience replay

---

## Examples

- The notebook visualizes learning performance.
- Trained models can be saved and replayed.
- Videos can be generated of the agent’s behavior.

---

## Troubleshooting

- If CUDA is not detected, training will proceed on CPU.
- Ensure `box2d` is installed properly for the environment.
- If video export fails, verify that `ffmpeg` is correctly installed via `imageio`.

---