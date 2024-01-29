"""module containing reinforcement learning policy class"""

import csv
from copy import deepcopy
import json
from math import sqrt
from os import listdir
from os.path import join
from random import randint, random
import sklearn
import time
import torch
from torch import cdist, unsqueeze

import data.dataloader as dataloader
from RL.Controller import RlController

from GP.GpModel import GpModel
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


class PolicyNetwork:
    """reinforcement learning policy"""

    def __init__(self,
                 gp_model: GpModel,
                 data_path:            str,
                 state_feature_count:  int = 9,
                 control_output_count: int = 3,
                 iterations:           int = 1000,
                 learning_rate:      float = 0.001):

        super().__init__()

        self.gp_model      = gp_model
        self.data_path     = data_path
        self.iterations    = iterations
        self.learning_rate = learning_rate

        # TODO make this configurable from someplace
        # TODO compute state_feature_count from this (how to know target dim?)
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

        self.losses = []

    def inverse_transform(self, scaler: str, data: torch.Tensor):
        """
        a copy of sklearn.preprocessing.MinMaxScaler.inverse_transform
        which handles tensors

        arguments:
            scaler: the key of the scaler in self.scalers, e.g. "velocities"
            data:   the tensor to inverse transform
        """

        data -= torch.tensor(self.scalers[scaler].min_,
                             device = self.device,
                             dtype  = self.dtype)

        data /= torch.tensor(self.scalers[scaler].scale_,
                             device = self.device,
                             dtype  = self.dtype)

        return data

    def get_random_states(self, batch_size):
        """
        fetches a batch of random states & next states from the trajectories
        in self.data_path

        A random state, including a random target, is fetched from each
        trajectory until the specified batch size is reached. If the number of
        trajectories is smaller than the desired batch size, the trajectories
        will be iterated further until the desired batch size is reached. The
        random state is taken from somewhere in the beginning 80% of each
        trajectory's data. The target is taken from somewhere between 20% after
        the initial state and the end of the trajectory.

        returns tuple of dicts (states, next_states)

        Each dict in the returned tuple comprises:
            keys: high-level features
            values: tensors of size (batch_size, number of low-level features)

        A high level feature is for example "velocities", which might contain
        two lower-level features for the x- and y-components respectively.
        Therefore, for a batch size of 100, states["velocities].shape and
        next_states["velocities"] would both be (100, 2).
        """

        # TODO also include real torque to compare against torque predictions

        csv_files = [file for file
                     in listdir(self.data_path)
                     if file.endswith('.csv')]

        list_of_states      = []

        actions = []

        while len(list_of_states) < batch_size:

            for csv_file in csv_files:

                file_path = join(self.data_path, csv_file)

                with open(file_path, "rb") as trajectory_file:
                    lines_in_file = sum(1 for _ in trajectory_file)

                last_line_in_file = lines_in_file - 2

                state_index = randint(a = 1,
                                      b = round(0.8 * last_line_in_file))

                next_state_index = state_index + 1

                target_index = randint(a =  (state_index
                                             + round(0.2 * last_line_in_file)),
                                       b = last_line_in_file)

                states = {}

                next_states = {}

                with open(file_path, 'r') as trajectory_file:

                    csv_reader = csv.DictReader(trajectory_file)

                    for index, row in enumerate(csv_reader):

                        if index in (state_index, next_state_index):

                            if index == state_index:

                                the_state = states

                                actions.append(
                                   [float(row[feature])
                                    for feature
                                    in dataloader.features["torques"]])

                            else:
                                the_state = next_states

                            for feature in dataloader.y_features:

                                the_state[feature] = \
                                          [float(row[feature])
                                           for feature
                                           in dataloader.features[feature]]

                        if index == target_index:

                            for the_state in (states, next_states):

                                the_state["target"] = \
                                    [float(row[feature])
                                     for feature
                                     in dataloader.features["ee_location"]]

                            break # quit looping this particular CSV file

                    assert "target" in states

                    list_of_states.append((states, next_states))

                if len(list_of_states) == batch_size:
                    break # quit looping over CSV files

        self.scalers["target"] = self.scalers["ee_location"]

        scaled_states      = {}
        scaled_next_states = {}

        for index, the_scaled_states in enumerate((scaled_states,
                                                   scaled_next_states)):

            for feature in ["target", *dataloader.y_features]:

                feature_data = [states[index][feature]
                                for states in list_of_states]

                scaled_feature_data = \
                    self.scalers[feature].transform(feature_data)

                the_scaled_states[feature] = \
                    torch.tensor(data   = scaled_feature_data,
                                 device = self.device,
                                 dtype  = self.dtype)

        actions = torch.tensor(data   = (self.scalers["torques"]
                                         .transform(actions)),
                               device = self.device,
                               dtype  = self.dtype)

        return scaled_states, actions, scaled_next_states

    def calculate_rewards(self, states):
        """calculate and return rewards for a batch of states"""

        targets = self.inverse_transform(scaler = "ee_location",
                                         data   = states["target"])

        ee_location = self.inverse_transform(scaler = "ee_location",
                                             data   = states["ee_location"])

        # vector from the predicted EE coordinates towards the goal
        ideal_vector_to_goal = (  states["target"]
                                - states["ee_location"])

        # TODO calculate the dot product between this and acceleration

        error_metric = {}

        for vector_metric in ("accelerations", "velocities"):

            vectors = \
                self.inverse_transform(scaler = vector_metric,
                                       data   = states[vector_metric])

            batch_size    = vectors.shape[0]
            feature_count = vectors.shape[1]

            x = ideal_vector_to_goal.reshape(batch_size, 1, feature_count)
            y = vectors.reshape(batch_size, feature_count, 1)

            # https://stackoverflow.com/a/65331075/837710
            dot_product = torch.matmul(x, y.clone()).squeeze((1, 2))

            norm = (  torch.norm(ideal_vector_to_goal, dim=1)
                    * torch.norm(vectors.clone(),      dim=1))

            # calculate the angle between the vectors
            error_metric[vector_metric] = torch.arccos(dot_product / norm)

            # TODO double-check whether this works and is sane & needed
            # compensate for angles > (1/2) * pi (they should be <= (1/2) pi)
            error_metric[vector_metric] = (  error_metric[vector_metric]
                                           % ((1/2) * torch.pi))

            # normalize to range (0, 1)
            error_metric[vector_metric] = \
                torch.div(error_metric[vector_metric],
                          ((1 / 2) * torch.pi))

        error_metric["euclidian_distance"] = \
            torch.norm(states["ee_location"] - states["target"], dim=1)

        # normalize Euclidian distance to range (0, 1)
        # x and y coordinates should be between -1 and 1
        # thus maximum possible distance is [-1, -1] → [1, 1] = 2 * sqrt(2)
        error_metric["euclidian_distance"] = \
            torch.div(error_metric["euclidian_distance"],
                      (2 * sqrt(2)))

        rewards = -sum((1 / len(error_metric)) * error
                       for error in error_metric.values())

        for metric_name, metric in error_metric.items():
          print(f"mean {metric_name} error: {metric.mean().item()}")

        self.losses.append({"reward": rewards.mean().item(),
                            **{metric_name: metric.tolist()
                               for metric_name, metric
                                in error_metric.items()}})

        with open("loss.json", "w") as loss_log:
            json.dump(self.losses, loss_log, indent = 1)

        return rewards

    def temporal_difference_learning(self):

        # TODO docstring

        ε = 0.9

        ε_decay = 0.99

        minimum_ε = 0.02

        optimizer = torch.optim.Adam(self.controller.parameters(),
                                     lr=self.learning_rate)

        huber_loss = torch.nn.HuberLoss()

        self.controller.train()

        for iteration in range(self.iterations):

            print(f"phase 1, iteration {iteration + 1}")

            print(f"epsilon: {ε:.1%}")

            optimizer.zero_grad()

            # i think only this line would be different for experience replay
            # instead, we would take states from the replay buffer
            states, ground_truth_actions, next_states = \
                self.get_random_states(batch_size = 200)

            actions = self.select_actions(states = states, ε = ε)

            ε = max(ε * ε_decay, minimum_ε)

            next_actions = self.select_actions(states = next_states, ε = 0)

            loss = huber_loss(input  = actions,
                              target = ground_truth_actions)

            print(f"loss: {loss.item()}\n")

            loss.backward()

            optimizer.step()

    def optimize_policy(self,
                        batch_size = 200,
                        ε = 0.9,
                        ε_decay = 0.99,
                        minimum_ε = 0.02):

        """optimize controller parameters"""

        # branch depending on whether to learn from:
          # next predicted state reward only
          # next predicted state reward vs next actual state reward
          # action vs actual action

        # experience_replay_buffer: dict? list?

        optimizer = torch.optim.Adam(self.controller.parameters(),
                                     lr=self.learning_rate)

        for iteration in range(self.iterations):

            print(f"phase 2, iteration {iteration + 1}")

            print(f"epsilon: {ε:.1%}")

            optimizer.zero_grad()

            states, _, _ = self.get_random_states(batch_size = batch_size)

            actions = self.select_actions(states = states, ε = ε)

            ε = max(ε * ε_decay, minimum_ε)

            next_states = self.predict_next_states(states, actions)
            # TODO record experiences to replay buffer, TODO then re-use

            rewards = self.calculate_rewards(next_states)

            loss = -rewards.mean()

            print(f"loss: {loss.item()}\n")

            # detach gradients to avoid accumulation from looping models
            for feature in dataloader.y_features:
                next_states[feature].detach()

            states  = next_states

            loss.backward()

            optimizer.step()

            torch.save(self.controller.state_dict(),
                       f"trained_models/RL-{iteration + 1}.pth")

    def select_actions(self, states, ε = 0):
        """
        given a batch of states, output a batch of actions

        currently alternates between exploiting & exploring with ε = 0.5
        """

        def print_value(title, tensor):
            """print a tensor's value for debugging"""

            print(f"{title}: {tensor.cpu().detach().numpy()}")

        controller_inputs = [states[feature]
                             for feature
                             in self.controller_input_features]

        controller_inputs = torch.cat(controller_inputs, dim=1)

        actions = self.controller(controller_inputs)

        if ε > 0:

            if random() <= (1 - ε):

                print("exploiting...")

            else:

                print("exploring...")

                reparameterization_ε = 0.1 # TODO add argument, disambiguate

                # add Gaussian noise for exploration (reparameterization trick)
                # https://sassafras13.github.io/ReparamTrick/
                actions = (  actions
                           + reparameterization_ε
                           * torch.randn_like(actions)
                           * actions.std())

        # for feature, value in states.items():
            # print_value(title = feature, tensor = value)

        return actions

    def predict_next_states(self, states, actions):
        """
        given a batch of states and a batch of actions,
        return the batch of next states
        """

        states["torques"] = actions

        for feature in dataloader.X_features: # check batch size matches
            assert states[feature].shape[0] == actions.shape[0]

        gp_inputs = [states[feature]
                     for feature in dataloader.X_features]

        gp_inputs = torch.cat(gp_inputs, dim=1)

        predictions = self.gp_model.predict(gp_inputs)

        # extract and update state with each high-level feature from output
        for feature in dataloader.y_features:

            (start_index,
             end_index) = \
                dataloader.get_feature_indices(
                    feature_names = dataloader.y_features,
                    query_feature = feature)

            states[feature] = predictions.mean[:, start_index : end_index]

        return states
