import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controller import RLController
from data import dataloader
from training.GPController import GPModel

class PolicyNetwork(nn.Module):
    def __init__(self, **params):
        super(PolicyNetwork, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)
        self.gp_model = gpmodel
            
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
