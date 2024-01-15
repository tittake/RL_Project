#! C:\Users\vaino\robot_projectWork\RL_Project\RLProject\Scripts\python.exe

from copy import deepcopy
import time
import os
import sys
import torch
import numpy as np

import os
from RL.configs import get_controller_params 

from data import dataloader
from RL.controller import RLController
from RL.utils import get_tensor, plot_policy
from GP_model.GPController import GPModel

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
        self.controller_params = get_controller_params()
        self.controller = self.get_controller()
        # self.gp_model = self.get_gp_model(**params)
        self.horizon = round(float(self.Tf) / float(self.dt))
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

    def get_controller(self):
        controller = RLController(**self.controller_params)
        RLmodel = controller.init_controller()
        return RLmodel
    
    def get_gp_model(self, **params):
        model = GPModel(**params)
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

        maxiter = self.rl_training_iter
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
                loss.backward()
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
        # state = np.concatenate([self.x_values.numpy(), self.y_values.numpy()], axis=1) # Current boom location
        # print("x size: ", self.x_values.shape)
        # print("y size: ",self.y_values.shape)
        # exit ()
        t1 = time.perf_counter() # time for the loss calulations 
        rewards = 0
        counter = 1
        print("STATE: ", self.joint_state)
        while counter <= self.horizon:
            action = self.controller(self.joint_state)
            print("ACTION: ", action)
            inputs = torch.tensor([np.concatenate([self.joint_state.detach().numpy(), action.detach().numpy()]).tolist()])
            print("INPUTS: ", inputs)
            predictions = self.gp_model.predict(inputs)
            print(predictions)
            # predict next state
            self.ee_location = predictions.mean[0, 0:2]
            self.joint_state = predictions.mean[0, 3:]
            print("EE: ", self.ee_location)
            print("JOINTTI: ", self.joint_state)
            # get reward
            reward = self.compute_reward(
                self.ee_location, target_state=self.target_ee_location
            )
            print("Reward: ", reward)
            rewards += reward
            counter += 1
        
        # mean_error = torch.mean(self.joint_state - self.target_ee_location)
        mean_error = 0
        t_elapsed = time.perf_counter() - t1
        print(f"Predictions completed, elapsed time: {t_elapsed:.2f}s")
        return rewards, mean_error
    
    
    def reset(self):
        # generate init/goal states
        self.initial_joint_state = torch.tensor([16.26891594, 23.19595227, -0.78539816], dtype=self.dtype)
        self.initial_ee_location = torch.tensor([2.86881573603111, 2.20217971813513], dtype=self.dtype)
        self.target_ee_location = torch.tensor([3.15479619153404, 0.948787661508362], dtype=self.dtype)
        # self.target_ee_location = np.concatenate([self.x_values.numpy(), self.y_values.numpy()], axis=1)
        
        # self.state = torch.zeros((self.horizon, self.state_dim), device=self.device, dtype=self.dtype)
        self.joint_state = self.initial_joint_state
        self.ee_location = deepcopy(self.initial_ee_location)
        # self.reward = torch.zeros((self.horizon), device=self.device, dtype=self.dtype)

        print(
            "Reset complete: {}..., goal: {}".format(
                self.joint_state, self.target_ee_location
            )
        )

    
    def compute_reward(self, state, target_state):
        # Compute Euclidean distance based reward
        distance = torch.norm(state - target_state)

        reward = -distance.item()

        return reward