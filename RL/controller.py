import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RLController:
    def __init__(self, **params):
        super(RLController, self).__init__()
        for key, value in params:
            setattr(self, key, value)

    def init_linear_controller(self):
        print("Initializing linear controller")
        self.model = torch.nn.Linear(
            self.state_dim, self.control_dim, dtype=self.dtype
        )
        self.controller = torch.nn.Sequential(self.linear_model)

        self.controller.predict = self.controller.forward
        return self.controller

