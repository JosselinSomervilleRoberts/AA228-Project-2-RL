# This file applies Q-learning to the data provided in data/
# under the format s,a,r,s' where s is the state, a is the action,
# r is the reward, and s' is the next state. The data is provided
# in the form of a csv file. The Q-learning algorithm is applied
# to the data and the resulting Q-table is saved in the form of
# a csv file.
# It also generates a plot of the Q-table for each state.
# And computes the optimal policy for each state, saves it in a
# csv file, and generates a plot of the policy for each state.

import numpy as np
from tqdm import tqdm
from utils import sample_outcome, sample_action, get_data_dict, get_data, save_policy
from scipy.spatial import cKDTree

class KernelSmoothing:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.kd_tree = cKDTree(X)
    
    def predict(self, x, factor=3):
        indices = self.kd_tree.query_ball_point(x, r=factor*self.bandwidth)
        X_neighbors = self.X[indices]
        Y_neighbors = self.Y[indices]
        weights = self.gaussian_kernel(x, X_neighbors, self.bandwidth)
        y_pred = np.sum(weights * Y_neighbors) / np.sum(weights)
        if np.isnan(y_pred):
            return self.predict(x, factor=factor*2)
        return y_pred
    
    def gaussian_kernel(self, x, X, bandwidth):
        diff = np.linalg.norm(X - x, axis=1)
        return np.exp(-0.5 * ((diff / bandwidth) ** 2)) / (np.sqrt(2 * np.pi) * bandwidth)


data = get_data("medium")
data_dict = get_data_dict("medium")

# Sorts the set of states and actiosn and create 4 dictionaries
# that map each state to an index and vice versa
# and each action to an index and vice versa
set_of_states = set(data['s'])
set_of_actions = set(data['a'])
sorted_set_of_states = sorted(set_of_states)
sorted_set_of_actions = sorted(set_of_actions)
action_to_index = {}
index_to_action = {}
for i in range(len(sorted_set_of_actions)):
    action = sorted_set_of_actions[i]
    action_to_index[action] = i
    index_to_action[i] = action

    

# Initialize Q-table
Q = np.zeros((50000, 7))
mask = -np.inf * np.ones((50000, 7))
for i in tqdm(range(data.shape[0])):
    # Get state, action, next state, and reward
    s = data.iloc[i, 0]
    a = data.iloc[i, 1]
    idx_s = s - 1
    idx_a = action_to_index[a]
    mask[idx_s, idx_a] = 0
# Prints dimensions of Q-table
print("Shape of Q-table: " + str(Q.shape))

# Initialize parameters
alpha = 0.1
gamma = 0.999
epsilon = 0.2
num_episodes = 50000
num_iter = 500

# Q-learning algorithm
# (It is adapted since we do not contain all couples of state-action)
for episode in tqdm(range(num_episodes)):
    # Initialize to random state
    s = np.random.choice(sorted_set_of_states)
    idx_s = s - 1
    # Loop until the end of the episode
    for _ in range(num_iter):
        idx_a = None
        # Choose action
        if np.random.rand() < epsilon:
            a = sample_action(data_dict, s)
            idx_a = action_to_index[a]
        else:
            idx_a = np.argmax(Q[idx_s, :] + mask[idx_s, :])
            a = index_to_action[idx_a]
        # Get next state and reward
        (s_prime, r) = sample_outcome(data_dict, s, a)
        # Update Q-table
        idx_s_prime = s_prime - 1
        if s_prime not in set_of_states:
            Q[idx_s, idx_a] += alpha * r
            break
        Q[idx_s, idx_a] += alpha * (r + gamma * np.max(Q[idx_s_prime, :] + mask[idx_s_prime, :]) - Q[idx_s, idx_a])
        # Update state
        s = s_prime
        idx_s = idx_s_prime

# Generates policy
policy = np.zeros(50000, dtype=int)

# Kernel smoothing
def state_to_pos_vel(s):
    pos = (s - 1) // 100
    vel = (s - 1) % 100
    return [pos, vel]
def state_action_to_pos_vel(s, a):
    pos = (s - 1) // 100
    vel = (s - 1) % 100
    return [pos, vel, a]
normalizer = np.array([500, 100, 7])
X = []
Y = []
for i in tqdm(range(data.shape[0])):
    # Get state, action, next state, and reward
    s = data.iloc[i, 0]
    a = data.iloc[i, 1]
    idx_s = s - 1
    idx_a = action_to_index[a]
    x = state_action_to_pos_vel(s, a) / normalizer
    X.append(x)
    Y.append(Q[idx_s, idx_a])
X = np.array(X)
Y = np.array(Y)
ks = KernelSmoothing(bandwidth=0.03)
ks.fit(X, Y)

for pos in tqdm(range(500)):
    for vel in range(100):
        for idx_a in range(7):
            idx_s = pos + 500 * vel
            s = 1 + idx_s
            a = index_to_action[idx_a]
            if mask[idx_s, idx_a] < 0:
                x = state_action_to_pos_vel(s, a) / normalizer
                Q[idx_s, idx_a] = round(ks.predict(x))

policy = np.zeros(50000, dtype=int)
for s in tqdm(range(1, 50001)):
    idx_s = s - 1
    policy[idx_s] = index_to_action[np.argmax(Q[idx_s, :])]

# Compute optimal policy for each state
save_policy(policy, "medium")

# Plot poliicy
# The X-axis is the velocity and the Y-axis is the position
# The color of each cell is the action to take
# The colorbar on the right shows the action
# A second plot shows the value function

# Create a figure
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 5))

# Add a subplot
ax = fig.add_subplot(1, 2, 1)

# Plot the policy (as a heatmap)
# Makes it square
ax.imshow(policy.reshape(500, 100), cmap="viridis", aspect=0.2)
# Set title
plt.title("Policy")

# Add labels
plt.xlabel("Velocity")
plt.ylabel("Position")

# Add a subplot
ax = fig.add_subplot(1, 2, 2)

# Plot the value function
ax.imshow(np.max(Q, axis=1).reshape(500, 100), cmap="viridis", aspect=0.2)
# Set title
plt.title("Value function")

# Add labels
plt.xlabel("Velocity")
plt.ylabel("Position")

# Show the plot
plt.show()

