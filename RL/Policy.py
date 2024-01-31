"""module containing reinforcement learning policy class"""

from copy import deepcopy
import csv
import json
from math import inf, sqrt
from os import listdir, mkdir
from os.path import dirname, isdir, join
from pathlib import Path
from random import randint, random
from typing import Callable, Literal

import torch
from tqdm import tqdm

import data.dataloader as dataloader
from RL.DQN import DQN

from GP.GpModel import GpModel

torch.autograd.set_detect_anomaly(True)


class RlPolicy:
    """reinforcement learning policy"""

    def __init__(self,
                 gp_model: GpModel,
                 data_path:            str,
                 state_feature_count:  int = 9,
                 control_output_count: int = 3,
                 saved_model_path:     str = None):

        super().__init__()

        self.gp_model  = gp_model
        self.data_path = data_path

        self.state_feature_count  = state_feature_count
        self.control_output_count = control_output_count

        # TODO make this configurable from someplace
        # TODO compute state_feature_count from this (how to know target dim?)
        self.input_features = ("target",
                               "joints",
                               "velocities",
                               "accelerations")

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        self.dtype = dataloader.dtype

        self.scalers = self.gp_model.scalers

        self.network = DQN(state_feature_count  = state_feature_count,
                           control_output_count = control_output_count)

        if saved_model_path is not None:

            state_dict = torch.load(saved_model_path)

            self.network.load_state_dict(state_dict)

        self.replay_buffer = []

        self.maximum_replay_buffer_size = 10**6

    def inverse_transform(self, scaler: str, data: torch.Tensor):
        """
        a copy of sklearn.preprocessing.MinMaxScaler.inverse_transform
        which handles tensors

        arguments:
            scaler: the key of the scaler in self.scalers, e.g. "velocities"
            data:   the tensor to inverse transform
        """

        data = data.clone()

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
        will be iterated further until the desired batch size is reached.

        returns tuple of dicts (states, next_states)

        Each dict in the returned tuple comprises:
            keys: high-level features
            values: tensors of size (batch_size, number of low-level features)

        A high level feature is for example "velocities", which might contain
        two lower-level features for the x- and y-components respectively.
        Therefore, for a batch size of 100, states["velocities].shape and
        next_states["velocities"] would both be (100, 2).
        """

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
                                      b = last_line_in_file - 1)

                next_state_index = state_index + 1

                target_index = randint(a = state_index + 1,
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

    def add_to_replay_buffer(self, states, next_states):

        """
        adds an experience to the experience replay buffer

        An experience consists of a state transition, here comprising
        `states` and `next_states`.
        """

        self.replay_buffer.append({"states":      states,
                                   "next_states": next_states})

        # truncate oldest experiences if buffer size exceeds maximum

        replay_buffer_length = 0

        index = len(self.replay_buffer) - 1

        first_feature = dataloader.y_features[0]

        while (    replay_buffer_length < self.maximum_replay_buffer_size
               and index > 0):

            replay_buffer_length += \
                len(self.replay_buffer[index]["states"][first_feature])

            index -= 1

        self.replay_buffer = self.replay_buffer[index:]

    def fetch_experiences_from_buffer(self, batch_size):

        """
        returns a random sample of `batch_size` from the experience replay
        buffer, or as many samples as possible if `batch_size` is greater than
        the number of experiences in the buffer
        """

        # generate batch_size unique indices like (batch_number, sample_number)

        indices = set()

        first_feature = dataloader.y_features[0]

        replay_buffer_length = sum(len(batch["states"][first_feature])
                                   for batch in self.replay_buffer)

        if replay_buffer_length < batch_size:

            # select all indices

            for batch_number, batch in enumerate(self.replay_buffer):

                batch_size = len(batch["states"][first_feature])

                for sample_number in range(batch_size):
                    indices.add((batch_number, sample_number))

        else: # sample batch_size random indices

            while len(indices) < batch_size:

                batch_number = randint(a = 0,
                                       b = len(self.replay_buffer) - 1)

                sample_number = \
                    randint(a = 0,
                            b = len(self.replay_buffer
                                    [batch_number]["states"][first_feature]
                                    ) - 1)

                indices.add((batch_number, sample_number))

        # construct & return dicts of {feature: tensor} for states, next_states

        states      = {}
        next_states = {}

        for feature in ["target", *dataloader.y_features]:

            states[feature] = \
                [self.replay_buffer
                 [batch_number]["states"][feature][sample_number]
                 for batch_number, sample_number in indices]

            next_states[feature] = \
                [self.replay_buffer
                 [batch_number]["next_states"][feature][sample_number]
                 for batch_number, sample_number in indices]

            states[feature] = \
                torch.stack(tensors = states[feature]).to(self.device)

            next_states[feature] = \
                torch.stack(tensors = next_states[feature]).to(self.device)

        return states, next_states

    def calculate_rewards(self, states):
        """calculate and return rewards for a batch of states"""

        targets = self.inverse_transform(scaler = "ee_location",
                                         data   = states["target"])

        ee_location = self.inverse_transform(scaler = "ee_location",
                                             data   = states["ee_location"])

        # vector from the predicted EE coordinates towards the goal
        ideal_vector_to_goal = targets - ee_location

        ideal_vector_to_goal = \
            self.inverse_transform(scaler = "ee_location",
                                   data   = ideal_vector_to_goal)

        error_metric = {}

        # for vector_metric in ("accelerations", "velocities"):
        # for vector_metric in ("accelerations", ):
        for vector_metric in ("velocities", ):

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

            # normalize to range (0, 1)
            error_metric[vector_metric] = \
                (error_metric[vector_metric] / torch.pi)

        error_metric["euclidian_distance"] = \
            torch.norm(states["target"] - states["ee_location"], dim=1)

        # normalize Euclidian distance to range (0, 1)
        # x and y coordinates should be between -1 and 1
        # thus maximum possible distance is [-1, -1] → [1, 1] = 2 * sqrt(2)
        error_metric["euclidian_distance"] = \
            (error_metric["euclidian_distance"] / sqrt(2))

        rewards = -sum((1 / len(error_metric)) * error
                       for error in error_metric.values())

        for metric_name, metric in error_metric.items():
            print(f"mean {metric_name} error: {metric.mean().item()}")

        return rewards

    def temporal_difference_learning(
        self,
        source:        Literal["ground_truth",
                               "experience_replay"],
        batch_size:    int   = 500,
        iterations:    int   = 1200,
        learning_rate: float = 0.001,
        ε:             float = 0.9,
        ε_decay:       float = 0.99,
        minimum_ε:     float = 0.02):

        """
        trains on temporal differences, either from ground truth trajectory
        data, or from experiences saved in the experience replay buffer

        A target DQN, separate from the main DQN, is instantiated in order to
        predict the Q-value action for the next state. The idea is to align the
        action at time t more closely to the optimal action at time t + 1.

        Huber Loss is used between the actions selected at the two time steps.

        ε and related parameters can be used for epsilon decay. ε represents
        the probability of exploration, rather than exploitation.
        """

        assert source in ("ground_truth", "experience_replay")

        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr = learning_rate)

        huber_loss = torch.nn.HuberLoss()

        self.network.train()

        print(f"temporal difference learning from "
              f"{source.replace('_', ' ')}")

        with tqdm(total = iterations) as progress_bar:

            for iteration in range(iterations):

                if (    source == "experience_replay"
                    and iteration % 10 == 0):

                    target_network = deepcopy(self.network)

                    target_network.eval()

                optimizer.zero_grad()

                if source == "ground_truth":

                    states, target_actions, next_states = \
                        self.get_random_states(batch_size = batch_size)

                elif source == "experience_replay":

                    states, next_states = \
                        self.fetch_experiences_from_buffer(
                            batch_size = batch_size)

                    target_actions = \
                        self.select_actions(network = target_network,
                                            states  = next_states,
                                            ε       = 0,
                                            quiet   = True)

                actions = self.select_actions(network = self.network,
                                              states  = states,
                                              ε       = ε,
                                              quiet   = True)

                ε = max(ε * ε_decay, minimum_ε)

                loss = huber_loss(input  = actions,
                                  target = target_actions)

                progress_bar.set_description(f"loss: {loss.item()}, "
                                             f"epsilon: {ε:.1%}")

                loss.backward()
                optimizer.step()

                progress_bar.update()

        print()

    def select_actions(self, network, states, ε = 0, quiet = False):
        """
        given a batch of states, output a batch of actions

        currently alternates between exploiting & exploring with ε = 0.5
        """

        def print_value(title, tensor):
            """print a tensor's value for debugging"""

            print(f"{title}: {tensor.cpu().detach().numpy()}")

        controller_inputs = [states[feature]
                             for feature
                             in self.input_features]

        controller_inputs = torch.cat(controller_inputs, dim=1)

        actions = network(controller_inputs)

        if ε > 0:

            if random() <= (1 - ε):

                if quiet is False:
                    print("exploiting...")

            else:

                if quiet is False:
                    print("exploring...")

                reparameterization_gain = 0.1 # TODO add argument

                # add Gaussian noise for exploration (reparameterization trick)
                # https://sassafras13.github.io/ReparamTrick/
                actions = (  actions
                           + reparameterization_gain
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
        # target should be left unmodified, as it should not be in y_features
        for feature in dataloader.y_features:

            (start_index,
             end_index) = \
                dataloader.get_feature_indices(
                    feature_names = dataloader.y_features,
                    query_feature = feature)

            states[feature] = predictions.mean[:, start_index : end_index]

        return states

    def train(self,
              batch_size:    int   = 400,
              iterations:    int   = 1200,
              learning_rate: float = 0.001,
              save_model_to: str   = None,
              ε:             float = 0.9,
              ε_decay:       float = 0.99,
              minimum_ε:     float = 0.02):

        """
        The training process is as follows:
          - temporal difference (TD) learning on the ground-truth trajectories
          - reward-based reinforcement learning
          - experience replay TD learning after every 10 batches
        """

        if save_model_to is None:

            print("WARNING: no path given for argument: `save_model_to` - "
                  "trained model will not be saved!\n")

        else:

            folder = dirname(save_model_to)

            if not isdir(folder):
                mkdir(folder)

            loss_log_path = join(folder,
                                 Path(save_model_to).stem
                                 + "-loss.json")

        self.temporal_difference_learning(
            source     = "ground_truth",
            batch_size = 500,
            iterations = 1200)

        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr = learning_rate)

        best_loss = inf

        loss_history = []

        for iteration in range(iterations):

            if (    iteration > 0
                and iteration % 10 == 0):

                self.temporal_difference_learning(
                    source     = "experience_replay",
                    batch_size = 500,
                    iterations = 1200,
                    ε          = ε)

            print(f"batched training, iteration {iteration + 1}")

            print(f"epsilon: {ε:.1%}")

            optimizer.zero_grad()

            states, _, _ = self.get_random_states(batch_size = batch_size)

            actions = self.select_actions(network = self.network,
                                          states  = states,
                                          ε       = ε)

            ε = max(ε * ε_decay, minimum_ε)

            next_states = self.predict_next_states(states, actions)

            rewards = self.calculate_rewards(next_states)

            loss = -rewards.mean()

            if save_model_to:

                if loss.item() < best_loss:

                    best_loss = loss.item()

                    torch.save(self.network.state_dict(), save_model_to)

                loss_history.append(loss.item())

                with open(loss_log_path, "w") as loss_log:
                    json.dump(loss_history, loss_log, indent = 1)

            print(f"loss: {loss.item()}\n")

            loss.backward()

            optimizer.step()

            # detach gradients to avoid accumulation from looping models
            for feature in dataloader.y_features:
                next_states[feature] = next_states[feature].detach()

            self.add_to_replay_buffer(states      = states,
                                      next_states = next_states)

            states = next_states

    def simulate_trajectory(self,
                            start_state:     dict,
                            target_location: dict,
                            iterations:      int            = 100,
                            online_learning: bool           = True
                            callback:        Callable[dict] = None):

        """
        simulate the trajectory from a given start state to a given target
        location, using the trained RL & GP models, optionally continuing
        training online

        A callback function can optionally be passed as an argument. The
        callback function should accept a single argument `state`, of type
        dict. Each time the state is updated  with a new prediction from the
        GP, it will be passed to the callback function.
        """

        def display_state(state):

            for feature in ("ee_location",
                            "target",
                            "velocities",
                            "accelerations"):

                scaler = feature if feature != "target" else "ee_location"

                inverse_scaled_feature_data = \
                    self.inverse_transform(scaler = scaler,
                                           data   = state[feature])[0]

                inverse_scaled_feature_data = \
                    inverse_scaled_feature_data.detach().cpu().numpy()

                print(f"{feature:<14}: {inverse_scaled_feature_data}")

            print()

        assert isinstance(target_location, torch.Tensor)

        if "target" not in start_state:
            start_state["target"] = target_location

        else:

            try:
                assert torch.equal(target_location, start_state["target"])

            except AssertionError:
                raise ValueError("state dict already includes target, "
                                 "but state['target'] != `target` argument.")

        initial_distance = \
            torch.cdist(torch.unsqueeze(start_state["ee_location"], dim=0),
                        torch.unsqueeze(target_location,            dim=0)
                        ).item()

        state = start_state

        display_state(state)

        # TODO test without online_learning

        if online_learning:

            self.network.train()

            optimizer = torch.optim.Adam(self.network.parameters(),
                                         lr = 0.01) # TODO add lr argument

        else:

            self.network.eval()

        for iteration in range(iterations):

            if online_learning:
                optimizer.zero_grad()

            action = self.select_actions(network = self.network,
                                         states  = state,
                                         ε       = 0)

            next_state = self.predict_next_states(state, action)

            display_state(next_state)

            distance = \
                torch.cdist(torch.unsqueeze(start_state["ee_location"], dim=0),
                            torch.unsqueeze(target_location,            dim=0)
                            ).item()

            print("percent distance covered: "
                  f"{(initial_distance - distance) / initial_distance:.1%}")

            if callback is not None:
                callback(state = deepcopy(next_state))

            if online_learning:

                reward = self.calculate_rewards(next_state)

                loss = -reward.mean()

                loss.backward()

                optimizer.step()

                # detach gradients to avoid accumulation from looping models
                for feature in dataloader.y_features:
                    next_state[feature] = next_state[feature].detach()

            print(f"loss: {loss.item()}\n")

            state = next_state

            # TODO stop when "close enough" to goal

    def simulate_random_trajectory(self,
                                   iterations:      int            = 100,
                                   online_learning: bool           = True
                                   callback:        Callable[dict] = None):

        """
        randomly sample a start state and target location from ground-truth
        data, then simulate the trajectory using the RL & GP models

        A callback function can optionally be passed as an argument. All
        arguments be passed to the more general `simulate_random_trajectory()`
        method above, so see its docstring for details.
        """

        if online_learning:
            self.network.train()
        else:
            self.network.eval()

        state, _, _ = self.get_random_states(batch_size = 1)

        target = state["target"]

        self.simulate_trajectory(start_state     = state,
                                 target_location = target,
                                 online_learning = True)
