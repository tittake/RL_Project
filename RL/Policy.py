"""module containing reinforcement learning policy class"""

from copy import deepcopy
from math import sqrt
import time
import torch
from torch import cdist, unsqueeze
import numpy as np

import data.dataloader as dataloader
from RL.Controller import RlController
from RL.utils import get_random_state

from GP.GpModel import GpModel
import matplotlib.pyplot as plt


class PolicyNetwork:
    """reinforcement learning policy"""

    def __init__(self,
                 gp_model: GpModel,
                 data_path:            str,
                 state_feature_count:  int = 7,
                 control_output_count: int = 3,
                 trials:               int = 100,
                 iterations:           int = 1000,
                 learning_rate:      float = 0.01):

        super().__init__()

        self.gp_model      = gp_model
        self.data_path     = data_path
        self.iterations    = iterations
        self.trials        = trials
        self.learning_rate = learning_rate

        # TODO make this configurable
        self.controller_input_features = ("joints",
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
                      unsqueeze(self.target_ee_location,   dim=0)
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
                      f"{self.target_ee_location.cpu().detach().numpy()}\n")

                optimizer.zero_grad()

                reward = self.calculate_step_reward()

                loss = -reward

                loss.backward()

                distance = \
                    cdist(unsqueeze(self.state["ee_location"], dim=0),
                          unsqueeze(self.target_ee_location,   dim=0)
                          ).cpu().detach().item()

                print(f"distance: {distance}")

                percent_distance_covered = (  (initial_distance - distance)
                                            / initial_distance)

                print("percent distance covered: "
                      f"{percent_distance_covered:.1%}\n")

                optimizer.step()

                optimization_log["loss"].append(loss.item())

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

        for feature, value in self.state.items():
            print_value(title = feature, tensor = value)

        controller_inputs = [self.state[feature]
                             for feature in self.controller_input_features]

        controller_inputs = \
            unsqueeze(torch.cat(controller_inputs), dim=0)

        self.state["torques"] = self.controller(controller_inputs)[0]

        print_value("action", self.state["torques"])

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

        print_value("current EE location", self.state['ee_location'])
        print_value("   goal EE location", self.target_ee_location)

        # vector from the predicted EE coordinates towards the goal
        ideal_vector_to_goal = (  self.target_ee_location
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
                  unsqueeze(self.target_ee_location,   dim=0))

        # normalize Euclidian distance to range (0, 1)
        # x and y coordinates should be between -1 and 1
        # thus maximum possible distance is [-1, -1] → [1, 1] = 2 * sqrt(2)
        error_metrics["euclidian_distance"] = \
            torch.div(error_metrics["euclidian_distance"],
                      (2 * sqrt(2)))

        reward = -sum((1 / len(error_metrics)) * error
                      for error in error_metrics.values())

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

            print(initial_state[feature])

            initial_state[feature] = \
                torch.tensor(self.scalers[feature]
                             .transform(initial_state[feature]),
                             device = self.device,
                             dtype  = self.dtype)[0]

        self.target_ee_location = initial_state["ee_location"].clone()

        # loop until initial and goal locations are distinct
        while torch.equal(self.target_ee_location,
                          initial_state["ee_location"]):

            random_state = get_random_state(self.data_path)

            self.target_ee_location = np.array([[random_state["boom_x"],
                                                 random_state["boom_y"]]
                                                ])

            self.target_ee_location = \
                torch.tensor(self.scalers["ee_location"]
                             .transform(self.target_ee_location),
                             device = self.device,
                             dtype  = self.dtype)[0]

        self.state = initial_state
