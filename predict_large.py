# This file tries to predict the next state and reward given a state and an action.
# It uses a neural network with 3 hidden layers of 100 neurons each.
# It runs on data/large.csv
# It splits the data into a training set and a test set.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

# Neural network
class Net(nn.Module):
    
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(61, 128)
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 61)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)

# Dataset
class MyDataset(Dataset):
        
            def __init__(self, X, y):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, index):
                return self.X[index], self.y[index]

# Load data
df = pd.read_csv('data/large.csv', header=0)
df.columns = ['state', 'action', 'reward', 'next_state']
X = df.drop(['next_state', 'reward'], axis=1)
y = df[['next_state', 'reward']]

# The features are:
# - the action (normalized)
# - for each digit (6 digits in total):
#   - if the digit = i (for i from 0 to 9)
def number_to_list(n):
    return [int(d) for d in str(n)]

def get_features(state, action):
    digits = number_to_list(int(state))
    features = [action] + [0] * 60
    for i in range(6):
        features[10*i + 1 + digits[i]] = 1
    return features

def prediction_to_state_reward(prediction):
    next_state = ''
    for i in range(6):
        next_state += str(np.argmax(prediction[0][10*i + 1:10*i + 11]))
    reward = prediction[0][0]
    return next_state, reward

npX = []
for i in X.iterrows():
    npX.append(get_features(i[1]['state'], i[1]['action']))
X = pd.DataFrame(npX)

# The target is:
# - The reward
# - for each digit (6 digits in total):
#   - if the digit = i (for i from 0 to 9)
def get_target(next_state, reward):
    digits = number_to_list(int(next_state))
    target = [reward] + [0] * 60
    for i in range(6):
        target[10*i + 1 + digits[i]] = 1
    return target

# Get target
npY = []
for i in y.iterrows():
    npY.append(get_target(i[1]['next_state'], i[1]['reward']))
y = pd.DataFrame(npY)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train.values).float()
y_test = torch.from_numpy(y_test.values).float()

# Create dataset
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

# Create dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# If a model is already saved, load it
# and test in on a few manual examples
if True:
    model = Net()
    model.load_state_dict(torch.load('models/large.pth'))
    model.eval()
    print('Model loaded')
    print('Test on a few manual examples')
    print('-----------------------------')
    print('State: 290411, Action: 1, Next state: 290411, Reward: -100')
    x = get_features(290411, 1)
    x = torch.tensor(x)
    # Normalize using the same scaler as the training set
    x = scaler.transform(x.reshape(1, -1))
    x = torch.tensor(x).float()
    print('Features: {}'.format(x))
    y = model(x).detach().numpy()
    print('Predicted next state and reward: {}'.format(y))
    next_state, reward = prediction_to_state_reward(y)
    print('Predicted next state: {}, predicted reward: {}'.format(next_state, reward))
    print('-----------------------------')

    # Sleep fro 30 seconds
    import time
    time.sleep(30)

# except Exception as e:
#     print("Error: {}".format(e))
#     print('No model found')

# Get features


# Create model
model = Net()

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train
# At each epoch, we train the model on the training set and evaluate it on the test set.
# We save the model if it performs better on the test set.
# And log the loss on the training set and the test set.
best_loss = 100000
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for i, (data, target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print('Epoch: {}, Train loss: {}, Test loss: {}'.format(epoch, train_loss, test_loss))
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'models/large.pth')





