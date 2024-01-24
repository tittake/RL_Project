"""module containing reinforcement learning policy class"""

from copy import deepcopy
import json
from math import sqrt
from random import random
import time
import torch
from torch import cdist, unsqueeze
import numpy as np
import csv

import data.dataloader as dataloader
from RL.Controller import RlController
from RL.utils import get_random_state

from GP.GpModel import GpModel
import matplotlib.pyplot as plt

RL_results = {"Joint state": [], 
              "Torque": [],
              "End-effector location": [],
              "Target": []}

class PolicyNetwork:
    """reinforcement learning policy"""

    def __init__(self,
                 gp_model: GpModel,
                 data_path:            str,
                 state_feature_count:  int = 9,
                 control_output_count: int = 3,
                 trials:               int = 100,
                 iterations:           int = 1000,
                 learning_rate:      float = 0.001):

        super().__init__()

        self.gp_model      = gp_model
        self.data_path     = data_path
        self.iterations    = iterations
        self.trials        = trials
        self.learning_rate = learning_rate

        # TODO make this configurable from someplace
        self.controller_input_features = ("target",
                                          "joints",
                                          "velocities",
                                          "accelerations")

        self.scalers = self.gp_model.scalers

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        self.dtype = dataloader.dtype

        self.controller = \
            RlController(state_feature_count  = state_feature_count,
                         control_output_count = control_output_count)

        self.rewards = []

    def optimize_policy(self):
        """optimize controller parameters"""

        trials = self.trials

        for trial in range(trials):

            self.reset()

            optimizer = torch.optim.Adam(self.controller.parameters(),
                                         lr=self.learning_rate)

            optimization_log = {"loss": [],
                                "time": []}

            initial_distance = \
                cdist(unsqueeze(self.state["ee_location"], dim=0),
                      unsqueeze(self.state["target"],      dim=0)
                      ).item()

            print(f"initial_distance: {initial_distance}\n")

            initial_ee_location = \
                self.state["ee_location"].clone().cpu().detach().numpy()

            start_model_training = time.perf_counter()

            self.controller.train()

            for i in range(self.iterations):

                print(f"trial {trial + 1}/{self.trials}, "
                      f"iteration {i + 1}/{self.iterations}\n")

                print(f"initial EE location: {initial_ee_location}")

                print("current EE location: "
                      f"{self.state['ee_location'].cpu().detach().numpy()}")

                print("target  EE location: "
                      f"{self.state['target'].cpu().detach().numpy()}\n")

                optimizer.zero_grad()

                reward = self.calculate_step_reward()

                loss = -reward

                loss.backward()

                distance = \
                    cdist(unsqueeze(self.state["ee_location"], dim=0),
                          unsqueeze(self.state["target"],      dim=0)
                          ).cpu().detach().item()

                print(f"distance: {distance}")

                percent_distance_covered = (  (initial_distance - distance)
                                            / initial_distance)

                print("percent distance covered: "
                      f"{percent_distance_covered:.1%}\n")

                optimizer.step()

                optimization_log["loss"].append(loss.item())

                RL_results["Torque"].append(self.state["torques"].cpu().detach().numpy())
                RL_results["Joint state"].append(self.state["joints"].cpu().detach().numpy())
                RL_results["End-effector location"].append(self.state["ee_location"].cpu().detach().numpy())
                RL_results["Target"].append(self.state["target"].cpu().detach().numpy())
            print(f"trial {trial + 1} distance covered: "
                  f"{percent_distance_covered:.1%}\n")

            if False:
                _, ax_loss = plt.subplots(figsize=(6, 4))

                ax_loss.plot(optimization_log["loss"], label='Training Loss')

                ax_loss.set_title('Training Loss Over Iterations')
                ax_loss.set_xlabel('Iteration')
                ax_loss.set_ylabel('Loss')

                ax_loss.legend()

                plt.show()

            torch.save(self.controller.state_dict(),
                       f"trained_models/RL-{trial + 1:03}.pth")
            

        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("training time: ", elapsed_model_training)
        if True:
            self.write_RL_results()

    def calculate_step_reward(self):
        """calculate and return reward for a single time step"""

        def print_value(title, tensor):
            """print a tensor's value for debugging"""

            print(f"{title}: {tensor.cpu().detach().numpy()}")

        # detach gradients to avoid accumulation due to looping models
        for feature in self.controller_input_features:
            self.state[feature] = \
                self.state[feature].to(self.device,
                                       dtype = self.dtype
                                       ).detach()

        exploiting = False

        # TODO doesn't make sense to choose action in calculate_step_reward()
        if random() <= 0.5:

            print("exploiting...")

            exploiting = True

            controller_inputs = [self.state[feature]
                                 for feature
                                 in self.controller_input_features]

            controller_inputs = \
                unsqueeze(torch.cat(controller_inputs), dim=0)

            self.state["torques"] = self.controller(controller_inputs)[0]

        else:

            print("exploring...")

            # random (normalized) torques between -1 and 1
            self.state["torques"] = -1 + 2 * torch.rand(size  = (3, ),
                                                        dtype = self.dtype
                                                        ).to(self.device)

        for feature, value in self.state.items():
            print_value(title = feature, tensor = value)

        print()

        gp_inputs = [self.state[feature]
                     for feature in dataloader.X_features]

        gp_inputs = unsqueeze(torch.cat(gp_inputs), dim=0)

        predictions = self.gp_model.predict(gp_inputs)

        for feature in dataloader.y_features:

            (start_index,
             end_index) = \
                dataloader.get_feature_indices(
                    feature_names = dataloader.y_features,
                    query_feature = feature)

            self.state[feature] = predictions.mean[0, start_index : end_index]

            self.state[feature].to(self.device, dtype = self.dtype)

        # vector from the predicted EE coordinates towards the goal
        ideal_vector_to_goal = (  self.state["target"]
                                - self.state["ee_location"])

        # TODO calculate the dot product between this and acceleration

        error_metrics = {}

        for vector_metric in ("accelerations", "velocities"):

            dot_product = torch.dot(ideal_vector_to_goal,
                                    self.state[vector_metric])

            dot_product = torch.dot(ideal_vector_to_goal,
                                    self.state[vector_metric]
                                    ).to(self.device, dtype = self.dtype)

            norm = (  torch.norm(ideal_vector_to_goal)
                    * torch.norm(self.state[vector_metric]))

            # calculate the angle between the vectors
            error_metrics[vector_metric] = \
                torch.arccos(torch.div(dot_product, norm))

            # normalize to range (0, 1)
            error_metrics[vector_metric] = \
                torch.div(error_metrics[vector_metric],
                          ((1 / 2) * torch.pi))

        error_metrics["euclidian_distance"] = \
            cdist(unsqueeze(self.state["ee_location"], dim=0),
                  unsqueeze(self.state["target"],      dim=0))

        # normalize Euclidian distance to range (0, 1)
        # x and y coordinates should be between -1 and 1
        # thus maximum possible distance is [-1, -1] → [1, 1] = 2 * sqrt(2)
        error_metrics["euclidian_distance"] = \
            torch.div(error_metrics["euclidian_distance"],
                      (2 * sqrt(2)))

        reward = -sum((1 / len(error_metrics)) * error
                      for error in error_metrics.values())

        if exploiting:

            self.rewards.append(reward.cpu().detach().item())

            with open("rewards.json", "w") as rewards_log:
                json.dump(self.rewards, rewards_log, indent = 1)

        print(f"reward: {reward.item()}")

        return reward

    def reset(self):
        """set initial & goal states"""

        random_state = get_random_state(self.data_path)

        initial_state = {}

        for feature in dataloader.y_features:

            initial_state[feature] = \
                np.array([[random_state[column]
                          for column in dataloader.features[feature]]])

            initial_state[feature] = \
                torch.tensor(self.scalers[feature]
                             .transform(initial_state[feature]),
                             device = self.device,
                             dtype  = self.dtype)[0]

        initial_state["target"] = initial_state["ee_location"].clone()

        # loop till initial & goal locations distinct (to avoid divide by zero)
        while torch.equal(initial_state["target"],
                          initial_state["ee_location"]):

            random_state = get_random_state(self.data_path)

            initial_state["target"] = np.array([[random_state["boom_x"],
                                                 random_state["boom_y"]]
                                                ])

            initial_state["target"]= \
                torch.tensor(self.scalers["ee_location"]
                             .transform(initial_state["target"]),
                             device = self.device,
                             dtype  = self.dtype)[0]

        self.state = initial_state

    def write_RL_results(self):
        
        # Move tensors to CPU and convert to NumPy arrays
        RL_results_np = {key: value.detach().numpy() if isinstance(value, torch.Tensor) else np.array(value) for key, value in RL_results.items()}

        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # Plot Joint state
        axs[0].plot(RL_results_np["Joint state"], marker='o', linestyle='-')
        axs[0].set_title('Joint State')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Joint State')

        # Plot Torque
        axs[1].plot(RL_results_np["Torque"], marker='o', linestyle='-')
        axs[1].set_title('Torque')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Torque')

        # Plot first End-effector location in red with 'o' markers and solid line
        axs[2].plot(RL_results_np["End-effector location"][:,0], marker='o', linestyle='-', color='red', label='End-effector Location 1')
        axs[2].plot(RL_results_np["End-effector location"][:,1], marker='x', linestyle='-', color='blue', label='End-effector Location 2')

        # Plot Target with '+' markers and dotted line in black
        axs[2].plot(RL_results_np["Target"][:,0], marker='o', linestyle='--', color='red', label='Target')
        axs[2].plot(RL_results_np["Target"][:,1],marker='x', linestyle='--', color='blue', label='Target')
        
        
        axs[2].set_title('End-effector Location')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('End-effector Location')

        data_length = len(RL_results["Joint state"])

        # Generate "Time" values from 0.1 to the length of the data
        time_values = [0.1 * i for i in range(1, data_length + 1)]

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

        print(RL_results_np)

        csv_file_path = 'stash/RL_results_polyfit.csv'
        # Lists to store column values
        time_list = []
        ee_x_list = []
        ee_y_list = []
        theta1_list = []
        theta2_list = []
        xt2_list = []
        torque1_list = []
        torque2_list = []
        torque3_list = []

        # Write data to lists
        time = 0.1
        for i in range(len(RL_results["Joint state"])):
            end_effector_x, end_effector_y = RL_results["End-effector location"][i]
            theta1, theta2, xt2 = RL_results["Joint state"][i]
            torque1, torque2, torque3 = RL_results["Torque"][i]

            end_effector_tensor = torch.from_numpy(np.array([end_effector_x, end_effector_y]))
            end_effector_tensor_2d = end_effector_tensor.view(1, -1)
            ee = self.scalers["ee_location"].inverse_transform(end_effector_tensor_2d)
            ee = ee[0]

            joint_tensor = torch.from_numpy(np.array([theta1, theta2, xt2]))
            joint_tensor = joint_tensor.view(1, -1)
            joint_tensor = self.scalers["joints"].inverse_transform(joint_tensor)
            joint_tensor = joint_tensor[0]

            torque_tensor = torch.from_numpy(np.array([torque1, torque2, torque3]))
            torque_tensor = torque_tensor.view(1, -1)
            torque_tensor = self.scalers["torques"].inverse_transform(torque_tensor)
            torque_tensor = torque_tensor[0]

            # Append values to lists
            time_list.append(time)
            ee_x_list.append(ee[0])
            ee_y_list.append(ee[1])
            theta1_list.append(joint_tensor[0])
            theta2_list.append(joint_tensor[1])
            xt2_list.append(joint_tensor[2])
            torque1_list.append(torque_tensor[0])
            torque2_list.append(torque_tensor[1])
            torque3_list.append(torque_tensor[2])
            time += 0.1

        # Choose the degree of the polynomial for fitting
        degree = 16  # You can adjust this based on your data

        # Fit a polynomial to the torque values
        torque1_list = np.polyfit(time_list, torque1_list, degree)
        poly_function = np.poly1d(torque1_list)
        torque1_list = poly_function(time_list)

        torque2_list = np.polyfit(time_list, torque2_list, degree)
        poly_function = np.poly1d(torque2_list)
        torque2_list = poly_function(time_list)

        torque3_list = np.polyfit(time_list, torque3_list, degree)
        poly_function = np.poly1d(torque3_list)
        torque3_list = poly_function(time_list)
            

        # Open the CSV file for writing
        with open(csv_file_path, 'w', newline='') as csvfile:
            # Create a CSV writer
            csv_writer = csv.writer(csvfile)

            # Write the header
            csv_writer.writerow(["time", "boom_x", "boom_y", "theta1", "theta2", "xt2", "fc1", "fc2", "fct2"])

            # Write the data rows
            for row in zip(time_list, ee_x_list, ee_y_list, theta1_list, theta2_list, xt2_list, torque1_list, torque2_list, torque3_list):
                csv_writer.writerow(row)

            print(f"Data written to {csv_file_path}")

            return
