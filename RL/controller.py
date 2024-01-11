import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RLController:
    def __init__(self, **params):
        super(RLController, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

    def init_linear_controller(self):
        print("Initializing linear controller")
        self.NNlayers = [64, 64]
        self.controller = torch.nn.Sequential(
            torch.nn.Linear(
                self.state_dim,
                self.NNlayers[0],
                device=self.device,
                dtype=self.dtype,
                bias=False,
            ),
            torch.nn.Linear(
                self.NNlayers[0],
                self.NNlayers[1],
                device=self.device,
                dtype=self.dtype,
                bias=False,
            ),
            torch.nn.Linear(
                self.NNlayers[1],
                self.control_dim,
                device=self.device,
                dtype=self.dtype,
                bias=False,
            ),
            torch.nn.Hardtanh(max_val=self.input_limit, min_val=-self.input_limit),
        )
        
        self.controller.predict = self.controller.forward

        return self.controller


