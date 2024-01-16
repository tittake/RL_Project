import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RLController:
    def __init__(self, **params):
        super(RLController, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

    def init_controller(self):

        print("Initializing controller")

        self.NNlayers = [64, 128, 64]

        self.controller = torch.nn.Sequential(
            torch.nn.Linear(
                self.state_dim,
                self.NNlayers[0],
                device=self.device,
                dtype=self.dtype,
                bias=False,
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                self.NNlayers[0],
                self.NNlayers[1],
                device=self.device,
                dtype=self.dtype,
                bias=False,
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                self.NNlayers[1],
                self.control_dim,
                device=self.device,
                dtype=self.dtype,
                bias=False,
            ),
            torch.nn.Tanh()
        )

        self.controller.predict = self.controller.forward

        return self.controller


