# Implements a DQN agent for the data in data/mdeium.csv
# State measurements are given by integers with 500 possible position values
# and 100 possible velocity values (50,000 possible state measurements).
# 1+pos+500*vel gives the integer corresponding to a state with position pos and velocity vel.
# There are 7 actions that represent different amounts of acceleration.
# This problem is undiscounted, and ends when the goal (the flag) is reached.
# Note that since the discrete state measurements are calculated after the simulation,
# the data in medium.csv does not quite satisfy the Markov property

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import namedtuple
from itertools import count
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Replay memory
class ReplayMemory(object):
    
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0
    
        def push(self, *args):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity
    
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
    
        def __len__(self):
            return len(self.memory)

# Neural network
class DQN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Agent
class DQNAgent(object):
        
        def __init__(self, input_size, output_size, batch_size=BATCH_SIZE, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, target_update=TARGET_UPDATE):
            self.input_size = input_size
            self.output_size = output_size
            self.batch_size = batch_size
            self.gamma = gamma
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay = eps_decay
            self.target_update = target_update
            self.steps_done = 0
            self.policy_net = DQN(self.input_size, self.output_size)
            self.target_net = DQN(self.input_size, self.output_size)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters())
            self.memory = ReplayMemory(10000)
        
        def select_action(self, state):
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.output_size)]], dtype=torch.long)
        
        def optimize_model(self):
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(self.batch_size)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

# Train the agent using the data in medium.csv
# This is batch reinforcement learning
# The data is under the form of a sequence of states, actions, rewards, and next states

def train_agent(agent, data_dict, num_episodes=1000):
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        # Starts with a rendom state between 1 and 50000
        state = torch.tensor([[random.randint(1, 50000)]], dtype=torch.long)
        for t in count():
            



# Run the agent
agent = DQNAgent(1, 5)
train_agent(agent)

# Test the agent
def test_agent(agent, num_episodes=100):
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        state = torch.tensor([[0]], dtype=torch.long)
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            reward = torch.tensor([0], dtype=torch.float)
            if action.item() == 0:
                reward = torch.tensor([1], dtype=torch.float)
            elif action.item() == 1:
                reward = torch.tensor([0.5], dtype=torch.float)
            elif action.item() == 3:
                reward = torch.tensor([-0.5], dtype=torch.float)
            elif action.item() == 4:
                reward = torch.tensor([-1], dtype=torch.float)
            next_state = torch.tensor([[state.item() + action.item() - 2]], dtype=torch.long)
            if next_state.item() > 50000:
                next_state = None
            # Move to the next state
            state = next_state
            if state is None:
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

test_agent(agent)

# Plot the results
def plot_results(agent, num_episodes=100):
    results = []
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        state = torch.tensor([[0]], dtype=torch.long)
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            reward = torch.tensor([0], dtype=torch.float)
            if action.item() == 0:
                reward = torch.tensor([1], dtype=torch.float)
            elif action.item() == 1:
                reward = torch.tensor([0.5], dtype=torch.float)
            elif action.item() == 3:
                reward = torch.tensor([-0.5], dtype=torch.float)
            elif action.item() == 4:
                reward = torch.tensor([-1], dtype=torch.float)
            next_state = torch.tensor([[state.item() + action.item() - 2]], dtype=torch.long)
            if next_state.item() > 50000:
                next_state = None
            # Move to the next state
            state = next_state
            if state is None:
                results.append(t)
                break
    plt.plot(results)
    plt.show()

plot_results(agent)

