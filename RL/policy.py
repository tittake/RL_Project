#! C:\Users\vaino\robot_projectWork\RL_Project\RLProject\Scripts\python.exe

from copy import deepcopy
import time
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RL.controller import RLController
from data import dataloader
from RL.utils import get_tensor, plot_policy

class PolicyNetwork:
    def __init__(self, **params):
        super(PolicyNetwork, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

        train_data = dataloader.load_training_data(self.train_path)
        test_data = dataloader.load_test_data(self.test_path)
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

    def compute_reward(self, state, target_state):
        # Compute Euclidean distance
        distance = torch.norm(state - target_state)

        reward = -distance.item()

        return reward

    def optimize_policy(self):
        """
        Optimize controller parameters
        """
        maxiter = self.training_iter #How many max iterations do we want
        trials = 10 # for testing purposes only

        for trial in range(trials):
            self.reset()

            self.set_optimizer()
            # reset and set optimizer over 10 trials
            # calculate & plot reward and mean error over max_iterations
            optimInfo = {"loss": [], "time": []}
            for i in range(maxiter):
                self.optimizer.zero_grad()
                reward, mean_error = self.opt_step_loss()
                loss = -reward
                loss.backward()
                print(
                    "Optimize Policy: Iter {}/{} - Loss: {:.3f}".format(
                        i + 1, maxiter, loss.item()
                    )
                )
                print("Mean error {:.5f}".format(mean_error))
                self.optimizer.step()
                # loss counter
                optimInfo["loss"].append(loss.item())

            # plot function, TODO some modifictaions that plots are correct
            plot_policy(
                controller=self.controller,
                model=self.gp_model_x_b,
                trial=trial + 1,
                reward=reward
            )
            print(
                "Controller's optimization: reward=%.3f."
                % (loss.item())
            )

    def opt_step_loss(self):
        """Calculate predictions and reward for one step and
            return results
            TODO: store actions and rewards to array(?)
        """
        state = deepcopy(self.gp_model) # end-effector location from GP model
        u = self.controller(state)
        t1 = time.perf_counter() # time for the loss calulations
        target_tensor = get_tensor(
            data=self.target_state
        )
        # predict next state
        #TODO make prediction function
        predictions = self.calc_predictions(self.gp_model)
        # TODO make proper fucntion for action choosing
        action = self.choose_action(state, predictions)
        # calculate reward and make counter for reward
        reward = self.compute_reward(state, action)
        rewards = rewards + reward
        state = action
        mean_error = torch.mean(state - self.target_state[0])
        time_elapsed = time.perf_counter() - t1
        print(f"elapsed time: {time_elapsed:.2f}s")
        return reward, mean_error

    def calc_predictions(gp_model):
        #To be implmented
        pass

    def choose_action(state, predictions):
        # claculate optimal action based on current state and predictions
        pass


    # yet to be implmented, planned to use for test out functions
    def train(self, train_data, num_epochs=1):
        for epoch in range(num_epochs):
            # Iterate over the training data
            for i in range(len(train_data['states'])):
                state = train_data['states'][i]
                action = train_data['actions'][i]
                reward = self.compute_reward(state)

                # Optimize the policy based on the current data point
                #self.optimize_policy(state, action, reward)

            print(f'Epoch {epoch + 1}/{num_epochs} completed.')

