import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count
from itertools import combinations_with_replacement

from tqdm import tqdm
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hamming_distance(int1, int2):
    # Convert integers to strings to access digits
    str1 = str(int1)
    str2 = str(int2)
    
    # Ensure both strings have the same length by zero-padding
    max_len = max(len(str1), len(str2))
    str1 = str1.zfill(max_len)
    str2 = str2.zfill(max_len)
    
    # Calculate the Hamming distance by summing absolute differences of digits
    hamming_dist = sum(abs(int(digit1) - int(digit2)) for digit1, digit2 in zip(str1, str2))
    
    return hamming_dist

# Sort the results (order does not matter)
dataset = pd.read_csv('123.csv', header=None)
dataset = dataset.iloc[:,2:5].to_numpy()
dataset = np.sort(dataset, axis=1)

dataset = 100 * dataset[:, 0] + 10 * dataset[:, 1] + dataset[: ,2]

# Define the environment
class LottoEnv:
    def __init__(self, dataset, history_len=5):
        temp = np.array(list(combinations_with_replacement(range(10), 3)))
        self.action_space = 100*temp[:,0] + 10*temp[:,1]+temp[:,2]
        self.n_actions = len(temp)

        self.history_len = history_len
        self.db = dataset
        self.reset()

    def sample_action(self):
        return np.random.choice(range(len(self.action_space)))

    def reset(self):
        self.idx = 1001
        return self.db[self.idx:self.idx+self.history_len].flatten()

    def step(self, action):
        self.idx = self.idx - 1
        action = self.action_space[action]

        terminated = self.db[self.idx] == action

        reward = (27 - hamming_distance(self.db[self.idx], action)) / 27

        while self.idx < 1001:
          self.idx = np.random.randint(1001, 7501)

        observation = self.db[self.idx:self.idx+self.history_len].flatten()

        return observation, reward, terminated, suc

# dump variable
env = LottoEnv(dataset, history_len=7) 

# Define Policy
class DecayEpsilonGreedy():

  def __init__(self, epsilon, factor=0):
    self._epsilon = epsilon
    self._factor = factor

  def get_action(self, qvalues):
    roll = np.random.rand()
    if(roll < self._epsilon):
      action = env.sample_action()
    else:
      action =  torch.argmax(qvalues).item()
      self._epsilon = max(self._epsilon - self._factor, 0.01)

    return action


MAX_CAPICITY = 1000
BATCH_SIZE = 32

class QNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(QNetwork, self).__init__()
    #self.fc1 = nn.Linear(input_size, hidden_size)
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
    self.fc3 = nn.Linear(hidden_size // 2, output_size)

  def forward(self, x):
    h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
    c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
    x = x.unsqueeze(1)
    out, _ = self.lstm(x, (h0,c0))

    x = out[:, -1, :]
    x = F.gelu(self.fc2(x))
    return self.fc3(x)


NUM_ACTS = env.n_actions;
NUM_PARAMS = env.history_len;

class Agent(object):

    def __init__(self, hidden_size, discount, epsilon):

        self.action_policy   = DecayEpsilonGreedy(epsilon, 9.99e-2)
        self.replay_buffer   = deque([], maxlen=MAX_CAPICITY)
        self.discount        = discount

        self.target_net   = QNetwork(NUM_PARAMS, hidden_size, NUM_ACTS).to(device)
        self.behavior_net = QNetwork(NUM_PARAMS, hidden_size, NUM_ACTS).to(device)

        self.target_net.load_state_dict(self.behavior_net.state_dict())
        self.target_net.eval()

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.behavior_net.parameters(), lr=1e-3)

    def sample_batch(self, batch_size):
        batch  = random.sample(self.replay_buffer, batch_size)
        state  = torch.stack ([sample[0] for sample in batch])
        action = torch.tensor([sample[1] for sample in batch], device=device)
        reward = torch.tensor([sample[2] for sample in batch], device=device)
        next_s = torch.stack ([sample[3] for sample in batch])
        done = torch.stack ([sample[4] for sample in batch])
        return state,action,reward,next_s,done

    def optimize(self):

        batch_size                            = min(len(self.replay_buffer), BATCH_SIZE)
        state, action, reward, next_s, done   = self.sample_batch(batch_size)

        behavior                             = self.behavior_net(state)

        Q_behavior                           = behavior[torch.arange(behavior.size(0)),action]

        Q_values                             = self.target_net(next_s)
        bootstrap, _                         = torch.max(Q_values, dim=1)

        targets = torch.where(done, reward, reward + self.discount * bootstrap)

        self.optimizer.zero_grad()
        loss = self.criterion(targets, Q_behavior)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update(self, state, action, reward, next_state, done):

        if reward is not None:
            self.replay_buffer.append([state, action, reward, next_state, done])

        return self.optimize()

    def step(self, state):
        qvalues = self.behavior_net(state)
        return self.action_policy.get_action(qvalues)
    
    def evaluate(self, idx_start, idx_stop):
        score = 0
        self.target_net.eval()
        for idx in range(idx_start+1,idx_stop):
           state = dataset[idx:idx+NUM_PARAMS].flatten()
           state = torch.tensor(state, device=device)
           qvalues = self.target_net(state)
           action = self.action_policy.get_action(qvalues)

           score += int(dataset[idx-1] == action)
        self.target_net.train()
        return score / (idx_stop - idx_start)


# Start the real train
agent = Agent(hidden_size=64, discount=0.99, epsilon=1.0)	# was 0.5

rewards = []
losses = []

n_games = 1000
step = 0

for game_num in range(n_games):

    s = torch.tensor(env.reset(), device=device)
    done = False

    reward_acc = 0
    loss_acc = 0

    curr_iter = 0

    while not done:
        action = agent.step(s)
        step += 1
        curr_iter += 1

        next_s, r, done, _ = env.step(action)

        r = torch.tensor(r, device=device)
        done = torch.tensor(done, device=device)
        next_s = torch.tensor(next_s, device=device)

        loss_acc += agent.update(s, action, r, next_s, done)

        s = next_s
        reward_acc += r.cpu()

        if step % 500 == 0:
            agent.target_net.load_state_dict(agent.behavior_net.state_dict())

        if done:
            break

    rewards.append(reward_acc)
    losses.append(loss_acc/curr_iter)

    if game_num % 50 == 0:
        validation = agent.evaluate(0,1000)
        print(f"Game: {game_num}, Reward: {rewards[-1]/50:.2f}, Loss: {losses[-1]:.2f}, Valiation accuracy: {validation:2.f}")


agent.target_net.load_state_dict(agent.behavior_net.state_dict())
agent.target_net.eval()

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(rewards, label='rewards')

plt.subplot(1, 2, 2)
plt.plot(losses, label='loss')

plt.savefig(f'result.png')
plt.close()

final_acc = agent.evaluate(0,7500)
print(f"Final accuracy on all dataset is {final_acc:.2f}")