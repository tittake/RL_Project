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
    def __init__(self, **params):
        super(RLController, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

    def init_linear_controller(self):
        print("Initializing linear controller")
        self.model = torch.nn.Linear(
            self.state_dim, self.control_dim, dtype=self.dtype
        )
        self.saturation = torch.nn.Hardtanh()
        self.controller = torch.nn.Sequential(self.linear_model)

        self.controller.predict = self.controller.forward
        return self.controller


