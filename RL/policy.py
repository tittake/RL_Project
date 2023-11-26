import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RL.controller import RLController
from data import dataloader
from training.GPController import GPModel

class PolicyNetwork:
    def __init__(self, **params):
        super(PolicyNetwork, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

        train_data = dataloader.load_training_data(self.train_path)
        test_data = dataloader.load_test_data(self.test_path)
        
                    
    #Not used, at least yet 
    def get_controller(self):
        controller = RLController(**self.controller_params)
        RLmodel = controller.init_controller()
        return RLmodel
    
    def set_optimizer(self):
        ## sets policy optimizer to Adam ##
        self.optimizer = torch.optim.Adam(
                [
                    {"params": self.controller.parameters()},
                ],
                lr = self.learning_rate
        )
