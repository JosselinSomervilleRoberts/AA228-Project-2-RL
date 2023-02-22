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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm



# Sorts the set of states and actiosn and create 4 dictionaries
# that map each state to an index and vice versa
# and each action to an index and vice versa
set_of_states = set(data['s'])
set_of_actions = set(data['a'])
sorted_set_of_states = sorted(set_of_states)
sorted_set_of_actions = sorted(set_of_actions)
state_to_index = {}
index_to_state = {}
action_to_index = {}
index_to_action = {}
for i in range(len(sorted_set_of_states)):
    state = sorted_set_of_states[i]
    state_to_index[state] = i
    index_to_state[i] = state
for i in range(len(sorted_set_of_actions)):
    action = sorted_set_of_actions[i]
    action_to_index[action] = i
    index_to_action[i] = action

    

# Initialize Q-table
Q = np.zeros((len(sorted_set_of_states), len(sorted_set_of_actions)))
# Prints dimensions of Q-table
print("Shape of Q-table: " + str(Q.shape))

# Initialize parameters
alpha = 0.1
gamma = 0.95
epsilon = 0.2
num_episodes = 1000
num_iter = 1000

# Q-learning algorithm
# (It is adapted since we do not contain all couples of state-action)
for episode in tqdm(range(num_episodes)):
    # Initialize to random state
    s = np.random.choice(sorted_set_of_states)
    idx_s = state_to_index[s]
    # Loop until the end of the episode
    for _ in range(num_iter):
        idx_a = None
        # Choose action
        if np.random.rand() < epsilon:
            a = sample_action(s)
            idx_a = action_to_index[a]
        else:
            idx_a = np.argmax(Q[idx_s, :])
            a = index_to_action[idx_a]
        # Get next state and reward
        (s_prime, r) = sample_outcome(s, a)
        # Update Q-table
        idx_s_prime = state_to_index[s_prime]
        Q[idx_s, idx_a] += alpha * (r + gamma * np.max(Q[idx_s_prime, :]) - Q[idx_s, idx_a])
        # Update state
        s = s_prime
        idx_s = idx_s_prime

# Save Q-table to csv file
Q_df = pd.DataFrame(Q)
Q_df.to_csv('data/Q.csv', index=False, header=False)

plot = False
if plot:
    # Plot Q-table for each state
    for s in range(Q.shape[0]):
        Q_s = Q[s, :]
        Q_s_df = pd.DataFrame(Q_s)
        Q_s_df.columns = ['Q']
        Q_s_df['a'] = Q_s_df.index
        plt.figure()
        sns.barplot(x='a', y='Q', data=Q_s_df)
        plt.title('Q-table for state ' + str(s))
        plt.xlabel('Action')
        plt.ylabel('Q')
        plt.savefig('plots/Q_table_state_' + str(s) + '.png')
        plt.close()

# Compute optimal policy for each state
policy = [index_to_action[idx] for idx in np.argmax(Q, axis=1)]

# Save policy to csv file
policy_df = pd.DataFrame(policy)
policy_df.to_csv('data/policy.csv', index=False, header=False)


# Evaluates policy and plots reward and accumulated reward
def evaluate_policy(policy, num_episodes, num_iter):
    """Evaluates a policy and plots reward and accumulated reward."""
    # Initialize parameters
    rewards = []
    accumulated_rewards = []
    # Loop over all episodes
    for episode in tqdm(range(num_episodes)):
        # Initialize to random state
        s = np.random.choice(sorted_set_of_states)
        # Initialize accumulated reward
        accumulated_reward = 0
        # Loop until the end of the episode
        for _ in range(num_iter):
            # Get action
            a = policy[state_to_index[s]]
            # Get next state and reward
            (s_prime, r) = sample_outcome(s, a)
            # Update accumulated reward
            accumulated_reward += r
            # Update state
            s = s_prime
        # Save reward and accumulated reward
        rewards.append(r)
        accumulated_rewards.append(accumulated_reward)
    # Plot reward and accumulated reward
    plt.figure()
    plt.plot(rewards)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('plots/reward.png')
    plt.close()
    plt.figure()
    plt.plot(accumulated_rewards)
    plt.title('Accumulated reward')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated reward')
    plt.savefig('plots/accumulated_reward.png')
    plt.close()

# Evaluate policy
evaluate_policy(policy, num_episodes=100, num_iter=100)
