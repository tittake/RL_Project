#! C:\Users\vaino\robot_projectWork\RL_Project\RLProject\Scripts\python.exe

from copy import deepcopy
import time
import torch
import numpy as np

from RL.configs import get_controller_params

from data import dataloader
from RL.controller import RLController
from RL.utils import generate_initial_values

from GP_model.GPController import GPModel
import matplotlib.pyplot as plt


class PolicyNetwork:

    def __init__(self, gp_model, iterations, trials, learning_rate = 0.1):

        super().__init__()

        self.gp_model = gp_model
        self.iterations = iterations
        self.trials = trials
        self.learning_rate = learning_rate

        self.joint_scaler = self.gp_model.joint_scaler
        self.torque_scaler = self.gp_model.torque_scaler
        self.ee_location_scaler = self.gp_model.ee_location_scaler

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        self.dtype = torch.double

        self.controller_params = get_controller_params()
        self.controller = self.get_controller()
        # self.gp_model = self.get_gp_model(**params)

    def get_controller(self):
        controller = RLController(**self.controller_params)
        RLmodel = controller.init_controller()
        return RLmodel

    def get_gp_model(self, **params):
        model = GPModel(**params)
        model.initialize_model()
        return model

    def optimize_policy(self):
        """
        Optimize controller parameters
        """

        trials = self.trials

        all_optim_data = {"all_optimizer_data": []}

        axs = None 
        
        for trial in range(trials):

            self.reset()

            initial_controller = deepcopy(self.controller)

            optimizer = torch.optim.Adam(self.controller.parameters(),
                                         lr=self.learning_rate)

            t_start = time.perf_counter()

            # reset and set optimizer over trials,
            # calculate and plot reward and mean error over max_iterations

            optimInfo = {"loss": [], "time": []}

            self.controller.train()

            start_model_training = time.perf_counter()


            for i in range(self.iterations):

                optimizer.zero_grad()

                print(f"optimize policy: iteration {i + 1} "
                      f"/ {self.iterations}")

                reward = self.calculate_step_reward()

                loss = -reward

                # print(loss)

                loss.backward()

                print()

                optimizer.step()

                optimInfo["loss"].append(loss.item())

            # TODO some modifications that plots are correct

            # plot_policy()

            print(
                "Controller's optimization: reward=%.3f."
                % (loss)
            )

            trial_save_info = {
                "optimInfo": optimInfo,
                "controller initial": initial_controller,
                "controller final": deepcopy(self.controller),
                "loss": loss,
                "trial": trial,
            }

            
            all_optim_data["all_optimizer_data"].append(trial_save_info)
            
            if False:
                _, ax_loss = plt.subplots(figsize=(6, 4))

                ax_loss.plot(optimInfo["loss"], label='Training Loss')

                ax_loss.set_title('Training Loss Over Iterations')
                ax_loss.set_xlabel('Iteration')
                ax_loss.set_ylabel('Loss')

                ax_loss.legend()

                plt.show()

        # TODO: better printing function
        # print("Optimized data: ", all_optim_data)
                
        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("Training time: ", elapsed_model_training)

            # Plot losses from all iterations for each trial at the end
        if isinstance(axs, np.ndarray):
            # Multiple subplots
            fig, axs = plt.subplots(len(all_optim_data["all_optimizer_data"]), 1, figsize=(8, 6 * len(all_optim_data["all_optimizer_data"])))
            fig.suptitle('Loss Subplots', y=0.92)
            fig.tight_layout(pad=3.0)

            for idx, trial_save_info in enumerate(all_optim_data["all_optimizer_data"]):
                ax = axs[idx]
                ax.set_title(f'Trial {trial_save_info["trial"]}')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                losses = trial_save_info["optimInfo"]["loss"]
                ax.plot(range(1, len(losses) + 1), losses)
        else:
            # Single subplot
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.suptitle('Loss Subplots', y=0.92)
            ax.set_title(f'Trial {all_optim_data["all_optimizer_data"][0]["trial"]}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            losses = all_optim_data["all_optimizer_data"][0]["optimInfo"]["loss"]
            ax.plot(range(1, len(losses) + 1), losses)

        plt.show()


    def calculate_step_reward(self):
        """calculate and return reward for a single time step"""

        self.joint_state = self.joint_state.to(self.device, dtype=self.dtype).detach()
        print(f"joint state: {self.joint_state.cpu().numpy()}")

        action = self.controller(self.joint_state)

        #print(f"action: {action.detach().numpy()}")
        print(f"action: {action.to('cpu').detach().numpy()}")


        inputs = torch.unsqueeze(torch.cat([self.joint_state, action]), dim=0)

        print(f"inputs: {inputs.to('cpu').detach().numpy()[0]}")

        # predict next state

        predictions = self.gp_model.predict(inputs)

        self.ee_location = predictions.mean[0, 0:2]
        self.joint_state = predictions.mean[0, 2:]

        print(f"joint state: {self.joint_state.to('cpu').detach().numpy()}")

        print("current EE location: "
              f"{self.ee_location.to('cpu').detach().numpy()}")

        print("   goal EE location: "
              f"{self.target_ee_location.to('cpu').detach().numpy()[0]}")

        self.ee_location = self.ee_location.to(self.device,
                                               dtype=torch.float64)

        self.target_ee_location = \
                self.target_ee_location.to(self.device, dtype=torch.float64)

        reward = \
            -torch.cdist(torch.unsqueeze(self.ee_location, dim=0),
                         torch.unsqueeze(self.target_ee_location, dim=0))

        print(f"reward: {reward.item()}")

        return reward

    def reset(self):

        """set initial & goal states"""
        
        generate_data_path = "trajectories/10Hz/all_joints"
        generated_values = generate_initial_values(generate_data_path)
        initial_ee = [generated_values["boom_x"], generated_values["boom_y"]]
        initial_joints = [generated_values["theta1"], generated_values["theta2"], generated_values["xt2"]]

        self.initial_joint_state = \
            torch.tensor(initial_joints,
                         dtype=self.dtype)

        self.initial_ee_location = \
            torch.tensor([initial_ee],
                         dtype=self.dtype)

        self.target_ee_location = \
            torch.tensor([[2.14813625484604,
                           3.5346459005167]],
                         dtype=self.dtype)

        self.initial_joint_state = \
            self.joint_scaler\
                .transform(self.initial_joint_state.view(1, -1).numpy())[0,:]
        
        self.initial_ee_location = \
            self.ee_location_scaler.transform(self.initial_ee_location)

        self.target_ee_location = \
            self.ee_location_scaler.transform(self.target_ee_location)

        self.initial_joint_state = torch.tensor(self.initial_joint_state)
        self.initial_ee_location = torch.tensor(self.initial_ee_location)
        self.target_ee_location = torch.tensor(self.target_ee_location)

        self.joint_state = deepcopy(self.initial_joint_state)
        self.ee_location = deepcopy(self.initial_ee_location)
