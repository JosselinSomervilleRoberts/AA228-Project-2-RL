# Implements Q-learning for data/large.csv file
# Uses heuristics to replace missing data (state-action pairs)
from scipy.spatial import cKDTree
from utils import get_data_dict, get_data, sample_outcome, save_policy
import numpy as np
import random
from tqdm import tqdm

# Suppress/hide the warning
np.seterr(invalid='ignore')

class NearestNeighborSmoothing:
    
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.kd_tree = cKDTree(X)

    def predict(self, x):
        distances, indices = self.kd_tree.query(x, k=self.k, workers=-1)
        # Uses a gaussian kernel to weight the neighbors
        weights = self.gaussian_kernel(distances)
        y_pred = np.sum(weights * self.Y[indices]) / np.sum(weights)
        return y_pred

    def gaussian_kernel(self, distances):
        return np.exp(-0.5 * ((distances / self.k) ** 2))

# Initialize Q-table
num_states = 312020
num_actions = 9
Q = np.zeros((num_states, num_actions))

# Get data
size = 'large'
data = get_data(size)
data_dict = get_data_dict(size)

# Initialize Q-table
mask = -np.inf * np.ones((num_states, num_actions))
for i in tqdm(range(data.shape[0]), desc="Generating mask"):
    # Get state, action, next state, and reward
    s = data.iloc[i, 0]
    a = data.iloc[i, 1]
    idx_s = s - 1
    idx_a = a - 1
    mask[idx_s, idx_a] = 0


def next_state(s, a):
    if a >= 5:
        return s
    elif a == 1:
        if s%100 == 20:
            return s
        if s%100 == 10:
            return s+1
        if s%10 == 4:
            return s+6
        elif s%10 == 0:
            return s+4
        else:
            return s+1
    elif a == 2:
        if s % 10 == 0:
            return s -  6
        if s % 100 == 11:
            return s - 1
        if s % 10 == 1:
            return s 
        return s - 1
    elif a == 3:
        if s%10000 // 100 == 1:
            return s
        if s%1000 // 100 == 0:
            return s - 600
        else:
            return s - 100
    elif a == 4:
        if s%10000 // 100 == 20:
            return s
        if s%1000 // 100 == 4:
            return s + 600
        else:
            return s + 100



# The heuistics for the reward is that if the state is in the data for a different action, then the reward is the same, otherwise it is 0.
# This only applies if action <= 4, otherwise the reward is 0.

state_non_zero_reward = {}
for row in data.iterrows():
    state = row[1]['s']
    action = row[1]['a']
    reward = row[1]['r']
    if action <= 4 and reward != 0:
        state_non_zero_reward[state] = reward

def get_reward(state, action):
    if action >= 5:
        return 0
    if state in state_non_zero_reward:
        return state_non_zero_reward[state]
    return 0

def get_outcome(state, action):
    # Tries to get the outcome from the data
    # If it is not in the data, it uses the heurestics
    if state in data_dict and action in data_dict[state]:
        return sample_outcome(data_dict, state, action)
    else:
        ns = next_state(state, action)
        if ns > num_states:
            ns = state
        if ns < 1:
            ns = state
        return ns, get_reward(state, action)

# Path to save Q-table
Q_path = 'Q_tables-large_q_learning.npy'

import os
if os.path.exists(Q_path):
    Q = np.load(Q_path)
    print("Q-table loaded from file")
else:
    print("Q-table not found, creating new one")
    # Hyperparameters
    alpha_max = 0.1
    gamma = 0.95
    epsilon = 0.2
    num_episodes = 500000
    num_iter = 5000

    # Q-learning algorithm
    # (It is adapted since we do not contain all couples of state-action)
    for episode in tqdm(range(num_episodes), desc='Episode'):
        # Update learning rate
        alpha = alpha_max * (1 - episode/num_episodes)
        # Initialize to random state
        s = 1 + random.randint(0, num_states-1)
        # Loop until the end of the episode
        for i in range(num_iter):
            # Choose action
            if random.random() < epsilon:
                a = 1 + random.randint(0, num_actions-1)
            else:
                a = 1 + np.argmax(Q[s-1])
            # Get next state
            s_prime, r = get_outcome(s, a)
            # Update Q-table
            Q[s-1][a-1] += alpha * (r + gamma * np.max(Q[s_prime-1]) - Q[s-1][a-1])
            # Update state
            s = s_prime

    # Save Q-table
    np.save(Q_path, Q)

# Get the policy
policy = np.argmax(Q, axis=1) + 1

# Save policy
save_policy(policy, 'large_q_learning')


X = []
action_to_pos = [0.2, 0.25, 0.3, 0.35, 0.796, 0.797, 0.798, 0.799, 0.8]
normalizer = np.array([0.1, 0.1, 0.05, 0.01, 0.02, 0.01, 1])
for i in tqdm(range(data.shape[0]), desc="Kernel initialization"):
    # Get state, action, next state, and reward
    s = data.iloc[i, 0]
    a = data.iloc[i, 1]
    x = np.array([float(d) for d in str(s)] + [action_to_pos[a-1]]) * normalizer
    X.append(x)
X = np.array(X)

# Kernel smoothing
Y = []
for i in tqdm(range(data.shape[0]), desc="Kernel initialization"):
    # Get state, action, next state, and reward
    s = data.iloc[i, 0]
    a = data.iloc[i, 1]
    idx_s = s - 1
    idx_a = a - 1
    Y.append(Q[idx_s, idx_a])
Y = np.array(Y)

ks = NearestNeighborSmoothing(k=100)
ks.fit(X, Y)


for idx_s in tqdm(range(num_states), desc="NN smoothing"):
    for idx_a in range(num_actions):
        s = 1 + idx_s
        str_s = str(s)
        str_s = "0" * (6 - len(str_s)) + str_s
        x = np.array([float(d) for d in str_s] + [action_to_pos[idx_a]]) * normalizer
        if mask[idx_s, idx_a] < 0:
            Q[idx_s, idx_a] = (Q[idx_s, idx_a] + 4.0 * ks.predict(x)) / 5.0

policy = np.argmax(Q, axis=1) + 1
save_policy(policy, 'large_q_learning_nn_smoothing')