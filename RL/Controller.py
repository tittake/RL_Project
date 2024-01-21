"""module containing PyTorch neural network used for RL Q-lookup"""

import torch
from torch import nn


class RlController(nn.Module):
    """PyTorch neural network used for deep RL Q-lookup"""

    def __init__(self, state_feature_count, control_output_count):

        super().__init__()

        device = torch.device("cuda:0"
                              if torch.cuda.is_available()
                              else "cpu")

        self.dtype = torch.double

        print("initializing controller...")

        self.nn_layers = [81, 243, 81]

        activation_function = nn.Tanh

        self.controller = nn.Sequential(
            nn.Linear(
                state_feature_count,
                self.nn_layers[0],
                device=device,
                dtype=self.dtype,
                bias=False,
            ),
            activation_function(),
            nn.Linear(
                self.nn_layers[0],
                self.nn_layers[1],
                device=device,
                dtype=self.dtype,
                bias=False,
            ),
            activation_function(),
            nn.Linear(
                self.nn_layers[1],
                control_output_count,
                device=device,
                dtype=self.dtype,
                bias=False,
            ),
            activation_function()
        )

    def forward(self, inputs):
        """feed forward"""
        return self.controller(inputs)
