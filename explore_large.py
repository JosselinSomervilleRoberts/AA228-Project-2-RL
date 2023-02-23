# Reads the CSV file data/non_zero_reward.csv
# and sorts it by state, and then by action.
# It then prints all sorted lines
# The colums are state, action, reward, and next state
# There is no header line

import csv
import numpy as np

# Read the CSV file and removes duplicates and lines that dont have 4 columns
lines = []
with open('data/non_zero_reward.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 4 and row not in lines:
            lines.append(row)

# Sort the lines by state, then by action
lines.sort(key=lambda x: (x[0], x[1]))

# Print the state, action, and the difference between the next state and the current state
# Prints it in green if the reward is positive
# Prints it in red if the reward is negative
for line in lines:
    reward = int(line[2])
    state = int(line[0])
    next_state = int(line[3])
    diff = next_state - state
    if reward > 0:
        print("\033[92m" + line[0] + " " + line[1] + " " + str(diff) + "\033[0m")
    elif reward < 0:
        print("\033[91m" + line[0] + " " + line[1] + " " + str(diff) + "\033[0m")
    else:
        print(line[0] + " " + line[1] + " " + str(diff))


# Reads data/large.csv
import pandas as pd
data = pd.read_csv('data/large.csv')
data.columns = ['state', 'action', 'reward', 'next_state']

# Sort data by state, then by action
data = data.sort_values(by=['state', 'action'])

# Remove duplicates and add a column count that counts the number of times a state, action, and next state appears
var_names = list(data.columns)
data = data.groupby(var_names).size().reset_index(name='count')

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

# For each line of data/large.csv
# prints the line that dont have the next state equal to next_state(state)
# print("\n\nThe following lines are wrong:")
# correct = 0
# incorrect = 0
# for index, row in data.iterrows():
#     # Checks if this is the biggest count for the pair state, action
#     b = row["count"] == data[(data["state"] == row["state"]) & (data["action"] == row["action"])]["count"].max()
#     if True:#row["action"] == 3 and b:
#         if next_state(row['state'], row['action']) == row['next_state']:
#             pass#print("\033[92m", np.array(row), "\033[0m")
#         else:
#             print("\033[91m", np.array(row), "\033[0m")
#     if next_state(row['state'], row['action']) == row['next_state']:
#         correct += row['count']
#     else:
#         incorrect += row['count']

# print("Correct: ", correct)
# print("Incorrect: ", incorrect)