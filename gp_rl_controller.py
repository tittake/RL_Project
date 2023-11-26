from training.GPController import GPModel
from RL.controller import RLController
import numpy as np
from config import configurations
from RL.policy import PolicyNetwork
import argparse


#Initial RL testing values
#x_boom = 2.1 y_boom = 3.5
#x_boom = 3.0 y_boom = 2.4


def get_configs():
    return

"""Collective controller for GP model and RL controller"""
def main(opts):

    #Initialize and train GP model
    gpmodel = GPModel()
    gpmodel.initialize_model(opts.train_path, opts.test_path, opts.num_tasks, opts.gp_inputs)
    gpmodel.train(opts.training_iter)
    gpmodel.plot_training_results()

    opts.gp_model = gpmodel
    policy_network = PolicyNetwork(**vars(opts))
    #policy_network.optimize_policy()
    

    #config = get_configs(opts)

    
if __name__ == "__main__":
    opts = argparse.Namespace()
    opts.train_path = "data/testing1_simple_10Hz.csv"
    opts.test_path = "data/training1_simple_10Hz.csv"
    opts.num_tasks = 2
    opts.gp_inputs =  6
    opts.training_iter = 100
    main(opts)