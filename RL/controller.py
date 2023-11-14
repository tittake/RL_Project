import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        return action

class RLController:
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action = self.policy_net(state_tensor)
        return action.detach().numpy()

    def train(self, states, actions, rewards):
        state_tensor = torch.FloatTensor(states)
        action_tensor = torch.FloatTensor(actions)
        reward_tensor = torch.FloatTensor(rewards)

        # Compute loss
        loss = -torch.mean(self.policy_net(state_tensor) * action_tensor * reward_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
