from training.GPController import GPModel
from RL.controller import RLController
import numpy as np
from config import configurations
from RL.policy import PolicyNetwork
import argparse
import torch

#Initial RL testing values
#x_boom = 2.1 y_boom = 3.5
#x_boom = 3.0 y_boom = 2.4


def get_configs():
    return

"""Collective controller for GP model and RL controller"""
def main(opts):

    #Initialize and train model or load pre-trained model
    gpmodel = GPModel(**vars(opts))

    gpmodel.plot_training_results()
        
    opts.gp_model = gpmodel

    policy_network = PolicyNetwork(**vars(opts))
    #policy_network.optimize_policy()
    

    #config = get_configs(opts)

#TODO: Separate GP and RL configs
if __name__ == "__main__":
    opts = argparse.Namespace()
    #opts.train_path = "data/some_chill_trajectories/trajectory14_100Hz.csv"
    #opts.test_path = "data/some_chill_trajectories/trajectory46_100Hz.csv"
    opts.train_path = "data/two-joint_trajectories_10Hz/trajectory2.csv" #Only use one trajectory test and train
    opts.test_path = "data/two-joint_trajectories_10Hz/trajectory3.csv" 
    opts.data_directory = 'data/two-joint_trajectories_10Hz' #Use a whole directory of data
    opts.num_tasks = 2
    opts.ard_num_dims = 6
    opts.training_iter = 150
    opts.train_GP = True
    opts.model_path = 'trained_models/two_joints_GP.pth'
    
    
    main(opts)