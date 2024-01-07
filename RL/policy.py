#! C:\Users\vaino\robot_projectWork\RL_Project\RLProject\Scripts\python.exe

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import time
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RL.controller import RLController
from data import dataloader
from training.GPController import GPModel
from utils import get_tensor, plot_policy

class PolicyNetwork:
    def __init__(self, **params):
        super(PolicyNetwork, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

        # Load training data, TODO: take this into use
        (
            self.x_values, 
            self.y_values,
        ) = dataloader.load_training_data(self.train_path)
        
        # test_data = dataloader.load_test_data(self.test_path)
        self.controller = self.get_controller()

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
    
    def optimize_policy(self):
        """
        Optimize controller parameters
        """
        maxiter = self.training_iter #How many max iterations do we want
        trials = self.trials
        
        all_optim_data = {"all_optimizer_data": []}
        
        for trial in range(trials):
            self.reset()
            initial_controller = deepcopy(self.controller)
            self.set_optimizer()
            t_start = time.perf_counter()
            # reset and set optimizer over trials and  calculate and plot reward and
            # mean error over the amount of max_iterations
            optimInfo = {"loss": [], "time": []}
            for i in range(maxiter):
                self.optimizer.zero_grad()
                reward, mean_error = self.calculate_step_loss()
                loss = -reward
                loss.backward() #TODO
                print(
                    "Optimize Policy: Iter {}/{} - Loss: {:.3f}".format(
                        i + 1, maxiter, loss.item()
                    )
                )
                print("Mean error {:.5f}".format(mean_error))
                self.optimizer.step()
                # loss counter
                t2 = time.perf_counter() - t_start
                optimInfo["loss"].append(loss.item())
                optimInfo["time"].append(t2)
                

            # plot function, TODO some modifictaions that plots are correct
            plot_policy(
                controller=self.controller,
                model=self.gp_model,
                trial=trial + 1,
                reward=reward
            )
            print(
                "Controller's optimization: reward=%.3f."
                % (loss.item())
            )
            
            trial_save_info = {
                "optimInfo": optimInfo,
                "controller initial": initial_controller,
                "controller final": deepcopy(self.controller),
                "mean_states": self.mean_states, #Don't exist yet
                "std_states": self.std_states, # Don't exist yet
            }
            
            all_optim_data["all_optimizer_data"].append(trial_save_info)

    def calculate_step_loss(self):
        """Calculate predictions and reward for one step and 
            return results
        """
        gpmodel = self.gpmodel
        state = deepcopy(self.y_values) # Boom location
        t1 = time.perf_counter() # time for the loss calulations 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        for i in range(self.horizon -1 ):
            X = [[0,0,0,1000,1000,1000]]
            X = torch.tensor(X, dtype=torch.double)
            X = X.to(device, dtype=torch.float64)
            predictions = gpmodel.predict(X)
            print(predictions.mean)
            # predict next state TODO
            next_state = (
                state
                + predictions.mean
                + torch.mul(predictions.stddev, self.randTensor[:, :, i]) # What is this
            )
            # get reward
            rewards = self.compute_reward(
                state, target_state=next_state
            )
            rew = rew + rewards
            state = next_state
        
        mean_error = torch.mean(state - self.target_state[0])
        t_elapsed = time.perf_counter() - t1
        print(f"Predictions completed, elapsed time: {t_elapsed:.2f}s")
        return rew, mean_error
    
    
    def reset(self):
        self.done = False

        # generate init/goal states TODO: discuss on how we should implement these functions
        initial_state = generate_init_state(
            is_det=self.is_deterministic_init,
            n_trajs=self.n_trajectories,
            initial_distr=self.initial_distr,
            x_lb=self.x_lb,
            x_ub=self.x_ub,
            state_dim=self.state_dim,
            default_init_state=self.init_state,
            device=self.device,
            dtype=self.dtype,
        )
        self.target_state = generate_goal(
            is_det=self.is_deterministic_goal,
            goal_distr=self.goal_distr,
            x_lb=self.x_lb,
            x_ub=self.x_ub,
            state_dim=self.state_dim,
            default_target_state=self.target_state,
        )

        self.state = torch.zeros(
            (self.horizon, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.state = initial_state
        self.reward = torch.zeros((self.horizon), device=self.device, dtype=self.dtype)

        self.obs_torch = initial_state
        print(
            "Reset complete: observation[0-10]: {}..., goal: {}".format(
                self.obs_torch[0:10], self.target_state
            )
        )
        return self.obs_torch

    
    def compute_reward(state, target_state):
        # Compute Euclidean distance based reward
        distance = torch.norm(state - target_state)

        reward = -distance.item()

        return reward

