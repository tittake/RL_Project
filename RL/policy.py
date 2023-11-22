import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from controller import RLController

class PolicyNetwork(nn.Module):
    def __init__(self, **params):
        super(PolicyNetwork, self).__init__()
        for key, value in params:
            setattr(self, key, value)
            
    def get_controller(self):
        controller = RLController(**self.controller_params)
        RLmodel = controller.init_controller()
        return RLmodel
    
    def set_optimazer(self):
        ## sets policy optimazer to Adam ##
        self.optimizer = torch.optim.Adam(
                [
                    {"params": self.controller.parameters()},
                ],
                lr = self.learning_rate
        )
