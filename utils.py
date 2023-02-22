import numpy as np
import pandas as pd
from tqdm import tqdm

def get_data_dict(model_size='small'):
    # Read data from csv file (The first row is the header)
    data = pd.read_csv('data/' + model_size + '.csv', header=0)
    data.columns = ['s', 'a', 'r', 's_prime']

    # Creates a dictionarray that for each state contains a dictionary of each action
    # and the corresponding next state and reward
    # This takes into account that we do not contain all couples of state-action
    data_dict = {}
    # Loop over all rows
    for i in tqdm(range(data.shape[0])):
        # Get state, action, next state, and reward
        s = data.iloc[i, 0]
        a = data.iloc[i, 1]
        s_prime = data.iloc[i, 3]
        r = data.iloc[i, 2]
        # If state is not in dictionary, add it
        if s not in data_dict:
            data_dict[s] = {}
        # Add action, next state, and reward to dictionary
        new_entry = (s_prime, r)
        if a not in data_dict[s]:
            data_dict[s][a] = {}
        if new_entry not in data_dict[s][a]:
            data_dict[s][a][new_entry] = 1
        else:
            data_dict[s][a][new_entry] += 1
    return data_dict


def get_available_actions(data_dict, s):
    """Returns a list of available actions for a given state."""
    return list(data_dict[s].keys())

def sample_action(data_dict, s):
    """Samples an action for a given state."""
    # Get all possible actions
    actions = get_available_actions(data_dict, s)
    # Sample an action
    action = np.random.choice(actions)
    return action

def sample_outcome(data_dict, s, a):
    """Samples an outcome for a given state and action."""
    # Get all possible outcomes
    outcomes = list(data_dict[s][a].keys())
    # Get the probabilities of each outcome
    probs = np.array(list(data_dict[s][a].values()))
    probs = probs / np.sum(probs)
    # Sample an outcome
    outcome = outcomes[np.random.choice(len(probs), p=probs)]
    return outcome

def save_policy(policy, model_size='small'):
    """Saves the policy to a csv file.
    The policy is an array of size (num_states), the output should be
    a file model_size.policy containing the action for each state on a new line.
    Converts the action to integer."""
    # Save policy to csv file
    policy_df = pd.DataFrame(policy, dtype=int)
    policy_df.to_csv('data/' + model_size + '.policy', index=False, header=False)