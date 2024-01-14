#! C:\Users\vaino\robot_projectWork\RL_Project\RLProject\Scripts\python.exe

from copy import deepcopy
import time
import os
import sys
import torch
import numpy as np

import os 

from data import dataloader
from RL.controller import RLController
from RL.utils import get_tensor, plot_policy

from GP_Model.GPController import GPModel

class PolicyNetwork:
    def __init__(self, **params):
        super(PolicyNetwork, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

        # Load training data
        (
            self.x_values, 
            self.y_values,
        ) = dataloader.load_training_data(self.train_path)
        
        # test_data = dataloader.load_test_data(self.test_path)
        self.controller = self.get_controller()
        self.gp_model = self.get_gp_model()
        self.horizon = round(self.Tf / self.dt)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

    def get_controller(self):
        controller = RLController(**self.controller_params)
        RLmodel = controller.init_controller()
        return RLmodel
    
    def get_gp_model(self):
        model = GPModel()
        model.initialize_model()
        return model

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(
                [{"params": self.controller.parameters()}, ],
                lr = self.learning_rate
        )

    
    def optimize_policy(self):
        """
        Optimize controller parameters
        """

        maxiter = self.training_iter
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
                #loss.backward()
                print(
                    "Optimize Policy: Iter {}/{} - Loss: {:.3f}".format(
                        i + 1, maxiter, loss
                    )
                )
                print("Mean error {:.5f}".format(mean_error))
                self.optimizer.step()
                # loss counter
                t2 = time.perf_counter() - t_start
                optimInfo["loss"].append(loss)
                optimInfo["time"].append(t2)
                

            # plot function, TODO some modifictaions that plots are correct
            # plot_policy()
            print(
                "Controller's optimization: reward=%.3f."
                % (loss)
            )
            
            trial_save_info = {
                "optimInfo": optimInfo,
                "controller initial": initial_controller,
                "controller final": deepcopy(self.controller),
                "mean error": mean_error,
                "Loss": loss,
                "Trial": trial,
            }
            
            all_optim_data["all_optimizer_data"].append(trial_save_info)
        
        # TODO: better printing function
        print("Optimized data: ", all_optim_data)


    def calculate_step_loss(self):
        """Calculate predictions and reward for one step and 
            return results
        """
        state = deepcopy(self.x_values, self.y_values) # Boom location
        t1 = time.perf_counter() # time for the loss calulations 
        rewards = 0
        counter = 1
        while counter <= self.horizon:
            X = [[0,0,0,1000,1000,1000]]
            X = torch.tensor(X, dtype=torch.double)
            X = X.to(self.device, dtype=torch.float64)
            predictions = self.gp_model.predict(X)
            print(predictions.mean)
            # predict next state
            next_state = (
                state
                + predictions.mean
            )
            # get reward
            reward = self.compute_reward(
                state, target_state=next_state
            )
            rewards = rewards + reward
            state = next_state
            counter += 1
        
        mean_error = torch.mean(state - self.target_state[0])
        t_elapsed = time.perf_counter() - t1
        print(f"Predictions completed, elapsed time: {t_elapsed:.2f}s")
        return rewards, mean_error
    
    
    def reset(self):
        # generate init/goal states
        initial_state = np.array([-0.5, 0])
        self.target_state = np.array(self.x_values, self.y_values)

        self.state = torch.zeros(
            (self.horizon, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.state = initial_state
        self.reward = torch.zeros((self.horizon), device=self.device, dtype=self.dtype)

        self.obs_torch = initial_state
        print(
            "Reset complete: observation[0-10]: {}..., goal: {}".format(
                self.obs_torch, self.target_state
            )
        )
        return self.obs_torch

    
    def compute_reward(self, state, target_state):
        # Compute Euclidean distance based reward
        distance = torch.norm(state - target_state)

        reward = -distance.item()

        return reward