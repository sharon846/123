import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse
import pandas as pd
from itertools import combinations_with_replacement

dataset = pd.read_csv('123.csv', header=None)
dataset = dataset.iloc[:,2:5].to_numpy()
dataset = np.sort(dataset, axis=1)

dataset = 100 * dataset[:, 0] + 10 * dataset[:, 1] + dataset[: ,2]

class DQN(nn.Module):
    def __init__(self, input_size, fc_num, hidden_size, output_size):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        fc_layers = []
        if fc_num > 1:
            fc_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (fc_num - 1)
        fc_layers.append(nn.Linear(hidden_size, output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)

        # We need to unsqueeze the sequence length dimension
        x = x.unsqueeze(1)

        # LSTM layer
        out, _ = self.lstm(x, (h0,c0))

        # Get the output from the last time step
        out = out[:, -1, :]

        out = self.fc(out)
        return out


# Define the environment
class LottoEnv:
    def __init__(self, dataset, history_len, patience_c):
        temp = np.array(list(combinations_with_replacement(range(10), 3)))
        self.action_space = 100*temp[:,0] + 10*temp[:,1]+temp[:,2]
        self.n_actions = len(temp)

        self.history_len = history_len
        self.max_steps = 2
        self.patience_const = patience_c
        self.db = dataset
        self.reset()

    def reset(self):
        self.idx = 7000
        self.steps = 0
        self.patience = self.patience_const * self.max_steps
        return np.ascontiguousarray(self.db[self.idx:self.idx+self.history_len][::-1])

    def step(self, action):
        self.idx = self.idx - 1
        action = self.action_space[action]

        suc = self.db[self.idx] == action

        if suc:
          self.steps += 1
        else:
          self.steps = 0

        reward = int(suc)

        if self.idx < 1001:
          self.idx = 7000

        observation = np.ascontiguousarray(self.db[self.idx:self.idx+self.history_len][::-1])
        terminated = self.steps == self.max_steps        # leave space for testing

        if terminated:
            self.patience -= 1
            if self.patience == 0:
                self.max_steps += 1
                print(f'max reward set to {self.max_steps}')

        return observation, reward, terminated, suc

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

parser = argparse.ArgumentParser(description='Net settings')
parser.add_argument('--lstm_hidden_size', type=int, help='LSTM hidden layer size')
parser.add_argument('--decay_param', type=float, help='Prize Decay parameter')
parser.add_argument('--history_len', type=int, help='States number')
parser.add_argument('--fc_num', type=int, help='FC layers number')
parser.add_argument('--patience', type=int, help='Patience for each reward')
parser.add_argument('--batch_size', type=int, help='batch size train on')

args = parser.parse_args()

id = f'L{args.lstm_hidden_size}_D{int(100*args.decay_param)}_H{args.history_len}_FC{args.fc_num}_P{args.patience}_B{args.batch_size}_R10'

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = args.batch_size
GAMMA = args.decay_param
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = LottoEnv(dataset, args.history_len, args.patience)

# Get number of actions from gym action space
n_actions = env.n_actions
# Get the number of state observations
state = env.reset()

policy_net = DQN(len(state), args.fc_num, args.lstm_hidden_size, n_actions).to(device)
target_net = DQN(len(state), args.fc_num, args.lstm_hidden_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = np.random.rand()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[np.random.randint(0, env.n_actions)]], device=device, dtype=torch.long)



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def evaluate_data(start, stop):
  score = 0
  for id in range(start,stop):
      state = np.ascontiguousarray(dataset[id:id+args.history_len][::-1])
      state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
      action = target_net(state).max(1).indices.view(1, 1).item()
      action = np.array([int(digit) for digit in str(action)])
      action = np.sort(action)
      action = int(''.join(map(str, action)))

      score += int(dataset[id-1] == action)
  return score / (stop - start)

def plot(epochs_num, rewards_in_epoch, accuracy_train, accuracy_validation):
  epochs = range(epochs_num)
  plt.plot(epochs, rewards_in_epoch, label='epoch acc', color='blue')
  plt.plot(epochs, accuracy_train, label='train acc', color='red')
  plt.plot(epochs, accuracy_validation, label='val acc', color='green')

  plt.xlabel('Epochs')
  plt.ylabel('Acc')
  plt.title('Acc Comparison Per Epochs')

  plt.legend()
  plt.savefig(f'../results/{id}_R10.png')
  plt.close()

def main():
  if torch.cuda.is_available():
      num_episodes = args.patience * 54         #2+..+10
  else:
      num_episodes = 50

  rewards_in_epoch = []
  accuracy_train = []
  accuracy_validation = []

  for i_episode in range(num_episodes):
      print('--------------------')
      # Initialize the environment and get it's state
      state = env.reset()
      state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
      
      rewards_in_epoch.append(0)
      for t in tqdm(count()):
          action = select_action(state)
          observation, reward, terminated, suc = env.step(action.item())
          reward = torch.tensor([reward], device=device)
          rewards_in_epoch[-1] += suc
          done = terminated

          if terminated:
              next_state = None
          else:
              next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

          # Store the transition in memory
          memory.push(state, action, next_state, reward)

          # Move to the next state
          state = next_state

          # Perform one step of the optimization (on the policy network)
          optimize_model()

          # Soft update of the target network's weights
          # θ′ ← τ θ + (1 −τ )θ′
          target_net_state_dict = target_net.state_dict()
          policy_net_state_dict = policy_net.state_dict()
          for key in policy_net_state_dict:
              target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
          target_net.load_state_dict(target_net_state_dict)

          if done:
              t = t + 1
              rewards_in_epoch[-1] = rewards_in_epoch[-1] / t
              accuracy_train.append(evaluate_data(1001,7000))
              accuracy_validation.append(evaluate_data(1,1001))

              print(f'epoch {i_episode} accuracy is {rewards_in_epoch[-1]:.5f}')
              print(f'train accuracy is {accuracy_train[-1]:.5f}')
              print(f'val accuracy is {accuracy_validation[-1]:.5f}')
              break
  print('--------------------')
  plot(num_episodes, rewards_in_epoch, accuracy_train, accuracy_validation)

main()
torch.save(target_net.state_dict(), f'../results/{id}_R10.pth')
state = np.ascontiguousarray(dataset[0:args.history_len][::-1])
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
print(target_net(state).max(1).indices.view(1, 1).item())
print('Complete')
