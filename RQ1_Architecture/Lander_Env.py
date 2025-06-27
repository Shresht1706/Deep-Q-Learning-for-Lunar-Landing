import os #for os
import random #for random numbers
import numpy as np #for arrays
import torch # to train agent w pytorch
import torch.nn as nn #neural network module
import torch.optim as optim # optimal module
import torch.nn.functional as F #functions pre made for training
import torch.autograd as autograd # for stochastic gradient descent
from torch.autograd import Variable #training
from collections import deque, namedtuple #training
import gymnasium as gym

env = gym.make("LunarLander-v3")
state_shape = env.observation_space.shape #Vector
state_size = env.observation_space.shape[0] #current state of env
number_actions = env.action_space.n #Number of actions

learning_rate = 5e-4 #derived from experimentation
minibatch_size = 100
gamma = 0.99 #discount factor
replay_buffer_size = int(1e5) #no. of experiences
tau = 1e-3 #Interpolation parameter

class ReplayMemory(object):

    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if exterior device(gpu) is present. uses that hardware to process
        self.capacity = capacity #total size of memory size
        self.memory = [ ]

    def push(self, event): #appends event and removes oldest event if memory is full
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        exp = random.sample(self.memory, k = batch_size)
        state = torch.from_numpy(np.vstack([e[0] for e in exp if e is not None])).float().to(self.device) #states converted to tensors and float values and send to gpu or cpu
        action = torch.from_numpy(np.vstack([e[1] for e in exp if e is not None])).long().to(self.device) #same as states but long integers
        rewards = torch.from_numpy(np.vstack([e[2] for e in exp if e is not None])).float().to(self.device)
        next_state = torch.from_numpy(np.vstack([e[3] for e in exp if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in exp if e is not None]).astype(np.uint8)).float().to(self.device)
        return state, next_state, action, rewards, dones

class Agent():

    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = None  # Will be assigned from Trainer
        self.target_qnetwork = None
        self.optimizer = None
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size: #self.memory.memory second memory is attribute. while self.memory is the instance of the memory class
                exp_local = self.memory.sample(minibatch_size) #samples 100 experiences from the memory
                self.learn(exp_local, gamma)

    def act(self, state, epsilon = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) #set as torch tensor and an extra variable to the vector to show batch number
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
            self.local_qnetwork.train()
            if random.random() > epsilon: #epsilon greedy action selection policy
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, exp, gamma):
        states,next_states, actions, rewards, dones = exp
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)






