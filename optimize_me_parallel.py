import multiprocessing as mp
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
import threading

# Suppress/hide the warning
np.seterr(invalid='ignore')

class KernelSmoothing:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.kd_tree = cKDTree(X)
    
    def predict(self, x, factor=3.0, count=0):
        indices = self.kd_tree.query_ball_point(x, r=factor*self.bandwidth)
        X_neighbors = self.X[indices]
        Y_neighbors = self.Y[indices]
        weights = self.gaussian_kernel(x, X_neighbors, self.bandwidth)
        y_pred = np.sum(weights * Y_neighbors) / np.sum(weights)
        if np.isnan(y_pred):
            if count > 10:
                print("Error: Kernel smoothing failed to converge.")
                return 1
            return self.predict(x, factor=factor*2, count=count+1)
        return y_pred
    
    def gaussian_kernel(self, x, X, bandwidth):
        diff = np.linalg.norm(X - x, axis=1)
        return np.exp(-0.5 * ((diff / bandwidth) ** 2)) / (np.sqrt(2 * np.pi) * bandwidth)


class Environment:

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.list_of_states = np.array(list(data_dict.keys()))
        self.reset()

    def reset(self):
        self.state = np.random.choice(self.list_of_states)
        return self.state

    def step(self, action):
        try:
            next_state, reward = sample_outcome(self.data_dict, self.state, action)
            self.state = next_state
            done = next_state not in self.list_of_states
            return next_state, reward, done, False
        except:
            return self.state, 0, True, True

class QLearner:
    def __init__(self, num_states, num_actions, data_dict, data, alpha=0.1, gamma=0.995, epsilon=0.2):
        self.Q = np.zeros((num_states, num_actions)) # Shared Q table
        self.mask = -np.inf * np.ones((num_states, num_actions)) # Shared mask
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.data_dict = data_dict
        self.init_mask(data)

    def init_mask(self, data):
        for i in tqdm(range(data.shape[0]), desc="Generating mask"):
            # Get state, action, next state, and reward
            s = data.iloc[i, 0]
            a = data.iloc[i, 1]
            idx_s = s - 1
            idx_a = a - 1
            self.mask[idx_s, idx_a] = 0
        
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            # Explore
            action = sample_action(self.data_dict, state)
        else:
            # Exploit
            idx_s = state - 1
            action = 1 + np.argmax(self.Q[idx_s] + self.mask[idx_s])
        return action
        
    def update_Q(self, state, action, reward, next_state, done=False, repet=0):
        # Q-learning update rule
        idx_a = action - 1
        idx_s = state - 1
        idx_s_prime = next_state - 1
        if done:
            if repet == 0:
                self.Q[idx_s, idx_a] += self.alpha * (reward - self.Q[idx_s, idx_a])
            else:
                self.Q[idx_s, idx_a] += self.alpha * (reward + self.gamma * np.max(self.Q[idx_s_prime, :]) - self.Q[idx_s, idx_a])
        else:
            if repet == 0:
                self.Q[idx_s, idx_a] += self.alpha * (reward + self.gamma * np.max(self.Q[idx_s_prime, :] + self.mask[idx_s_prime, :]) - self.Q[idx_s, idx_a])
            else:
                self.Q[idx_s, idx_a] += self.alpha * (reward + self.gamma * np.max(self.Q[idx_s_prime, :]) - self.Q[idx_s, idx_a])
        
def run_episode(learner, env, thread_num, num_episodes, pbar=None):
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = learner.choose_action(state)
            next_state, reward, done, crashed = env.step(action)
            if not crashed:
                learner.update_Q(state, action, reward, next_state, done=done, repet=0)
                state = next_state
        if pbar is not None:
            pbar.update()
            
def parallel_Q_learning(learner, env, num_threads, num_episodes):
    threads = []
    # Create a tqdm progress bar
    pbar = tqdm(total=num_threads*num_episodes)
    for i in range(num_threads):
        t = threading.Thread(target=run_episode, args=(learner, env, i, num_episodes, pbar))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Close the progress bar
    pbar.close()
    return learner.Q

data = get_data("medium")
data_dict = get_data_dict("medium")
env = Environment(data_dict)
learner = QLearner(50000, 7, data_dict, data, alpha=0.1, gamma=0.995, epsilon=0.2)

#run_episode(learner, env, 0, 1000)

# Run parallel Q-learning
Q = parallel_Q_learning(learner, env, 32, 2000)

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
for i in tqdm(range(data.shape[0]), desc="Kernel initialization"):
    # Get state, action, next state, and reward
    s = data.iloc[i, 0]
    a = data.iloc[i, 1]
    x = state_action_to_pos_vel(s, a) / normalizer
    X.append(x)
X = np.array(X)


# Generates policy
policy = np.zeros(50000, dtype=int)

# Kernel smoothing
Y = []
for i in tqdm(range(data.shape[0]), desc="Kernel initialization"):
    # Get state, action, next state, and reward
    s = data.iloc[i, 0]
    a = data.iloc[i, 1]
    idx_s = s - 1
    idx_a = a - 1
    Y.append(learner.Q[idx_s, idx_a])
Y = np.array(Y)

ks = KernelSmoothing(bandwidth=0.03)
ks.fit(X, Y)

for pos in tqdm(range(500), desc="Kernel smoothing"):
    for vel in range(100):
        for idx_a in range(7):
            idx_s = pos + 500 * vel
            s = 1 + idx_s
            a = 1 + idx_a
            if learner.mask[idx_s, idx_a] < 0:
                x = state_action_to_pos_vel(s, a) / normalizer
                learner.Q[idx_s, idx_a] = ks.predict(x)
            else:
                x = state_action_to_pos_vel(s, a) / normalizer
                learner.Q[idx_s, idx_a] = (learner.Q[idx_s, idx_a] + 2.0 * ks.predict(x)) / 3.0

policy = np.zeros(50000, dtype=int)
for s in tqdm(range(1, 50001)):
    idx_s = s - 1
    policy[idx_s] = 1 + np.argmax(learner.Q[idx_s, :])

# Compute optimal policy for each state
save_policy(policy, "medium_thread")

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
ax.imshow(np.max(learner.Q, axis=1).reshape(500, 100), cmap="viridis", aspect=0.2)
# Set title
plt.title("Value function")

# Add labels
plt.xlabel("Velocity")
plt.ylabel("Position")

# Show the plot
plt.show()