import numpy as np
import torch
import yaml

from training.GPController import GPModel
from RL.controller import RLController
from RL.policy import PolicyNetwork

#Initial RL testing values
#x_boom = 2.1 y_boom = 3.5
#x_boom = 3.0 y_boom = 2.4


def get_configs():
    return

"""Collective controller for GP model and RL controller"""
def main(configuration):

    #Initialize and train model or load pre-trained model
    gpmodel = GPModel(**configuration)

    gpmodel.plot_training_results()
        
    configuration.gp_model = gpmodel

    print(gpmodel.predict([0, 0, 0, 1, 1, 1]))

    policy_network = PolicyNetwork(**configuration)

    #policy_network.optimize_policy()

#TODO: Separate GP and RL configs
if __name__ == "__main__":

    with open("configuration.yaml") as configuration_file:

        configuration = yaml.load(configuration_file,
                                  Loader = yaml.Loader)
    
    main(configuration)
