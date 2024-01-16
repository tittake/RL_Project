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
import matplotlib.pyplot as plt

class PolicyNetwork:
    def __init__(self, **params):
        super(PolicyNetwork, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

        # Load training data
        (
            self.x_values, 
            self.y_values,
            self.X_scaler,
            self.y_scaler
        ) = dataloader.load_training_data(self.train_path)
        
        # test_data = dataloader.load_test_data(self.test_path)
        (
            self.x_values, 
            self.y_values,
            self.x_test_values,
            self.y_test_values,
            self.X_scaler,
            self.y_scaler
        ) = dataloader.load_data_directory(self.data_directory)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        self.x_values = self.x_values.to(self.device, dtype=self.dtype)
        self.y_values = self.x_values.to(self.device, dtype=self.dtype)
        self.x_test_values = self.x_values.to(self.device, dtype=self.dtype)
        self.y_test_values = self.x_values.to(self.device, dtype=self.dtype)
        
        self.controller_params = get_controller_params()
        self.controller = self.get_controller()
        # self.gp_model = self.get_gp_model(**params)
        self.horizon = round(float(self.Tf) / float(self.dt))
        

    def get_controller(self):
        controller = RLController(**self.controller_params)
        RLmodel = controller.init_controller()
        return RLmodel
    
    def get_gp_model(self, **params):
        model = GPModel(**params)
        model.initialize_model()
        return model

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.rl_learning_rate)

    
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
            #rewards = 0
            self.controller.train()
            
            for i in range(maxiter):
                self.optimizer.zero_grad()
                reward, mean_error = self.calculate_step_loss()
                #rewards += reward
                loss = -torch.tensor(reward, requires_grad=True)
                loss.backward()
                print("Loss: ", loss)
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
        #print("Optimized data: ", all_optim_data)
        _, ax_loss = plt.subplots(figsize=(6, 4))
        ax_loss.plot([tensor.detach().numpy() for tensor in optimInfo["loss"]], label='Training Loss')
        ax_loss.set_title('Training Loss Over Iterations')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        plt.show()


    def calculate_step_loss(self):
        """Calculate predictions and reward for one step and 
            return results
        """
        # state = np.concatenate([self.x_values.numpy(), self.y_values.numpy()], axis=1) # Current boom location
        # print("x size: ", self.x_values.shape)
        # print("y size: ",self.y_values.shape)
        # exit ()
        t1 = time.perf_counter() # time for the loss calulations 
        counter = 1

        self.joint_state = self.joint_state.to(self.device, dtype=self.dtype)

        #while counter <= self.horizon:
            
        action = self.controller(self.joint_state)
        #print("Action: ", action)
        #print("EE_Loc", self.ee_location)
        #print("ee_target", self.target_ee_location)
        inputs = torch.tensor([np.concatenate([self.joint_state.detach().cpu().numpy(), action.detach().cpu().numpy()]).tolist()])
        predictions = self.gp_model.predict(inputs)
        
        # predict next state
        self.ee_location = predictions.mean[0, 0:2]
        self.joint_state = predictions.mean[0, 2:]

        self.ee_location = self.ee_location.to(self.device, dtype=torch.float64)
        self.target_ee_location = self.target_ee_location.to(self.device, dtype=torch.float64)
        
        # get reward
        reward = self.compute_reward(
            ee_location=self.ee_location, target_ee_location=self.target_ee_location
        )
        print("Reward: ", reward)
        
        counter += 1

        mean_error = torch.norm(self.ee_location - self.target_ee_location)
        #mean_error = 0
        t_elapsed = time.perf_counter() - t1
        print(f"Predictions completed, elapsed time: {t_elapsed:.2f}s")
        return reward, mean_error
    
    
    def reset(self):
        # generate init/goal states

        self.initial_joint_state = torch.tensor([0.203369397750485,0.833537660540792,0.20976320774321], dtype=self.dtype)
        self.initial_ee_location = torch.tensor([ [2.34771798310625,3.24988861728872]], dtype=self.dtype)
        self.target_ee_location = torch.tensor([[1.62151587273472,4.0491973610912]], dtype=self.dtype)
        
        #initial torques 
        initial_situation = torch.tensor([[0.203369397750485,0.833537660540792,0.20976320774321, 41415.8512113725,-52444.0587430585,4870.27172582177]], dtype=self.dtype)
        self.initial_joint_state = self.X_scaler.fit_transform(initial_situation)[0,0:3]

        self.initial_ee_location = self.y_scaler.fit_transform(self.initial_ee_location)
        self.target_ee_location = self.y_scaler.fit_transform(self.target_ee_location)
        
        self.initial_joint_state = torch.tensor(self.initial_joint_state)
        self.initial_ee_location = torch.tensor(self.initial_ee_location)
        self.target_ee_location = torch.tensor(self.target_ee_location)
        # self.target_ee_location = np.concatenate([self.x_values.numpy(), self.y_values.numpy()], axis=1)
        
        # self.state = torch.zeros((self.horizon, self.state_dim), device=self.device, dtype=self.dtype)
        self.joint_state = deepcopy(self.initial_joint_state)
        self.ee_location = deepcopy(self.initial_ee_location)
        # self.reward = torch.zeros((self.horizon), device=self.device, dtype=self.dtype)

        print(
            "Reset complete: {}..., goal: {}".format(
                self.joint_state, self.target_ee_location
            )
        )

    
    def compute_reward(self, ee_location, target_ee_location):
        # Compute Euclidean distance based reward
        
        distance = torch.norm(ee_location - target_ee_location)

        reward = -distance.item()

        return reward