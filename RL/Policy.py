"""module containing reinforcement learning policy class"""

from copy import deepcopy
import time
import torch
import numpy as np

from data.dataloader import features, X_features, y_features
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

        self.scalers = self.gp_model.scalers

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        # TODO find a way to make self.dtype consistent with dataloader dtype
        # dataloader has it hard-coded as torch.double...
        # but many statements below cast to torch.float64...
        self.dtype = torch.double

        self.controller = \
            RlController(state_feature_count  = state_feature_count,
                         control_output_count = control_output_count)

    def optimize_policy(self):
        """optimize controller parameters"""

        trials = self.trials

        all_optim_data = {"all_optimizer_data": []}

        axs = None

        for trial in range(trials):

            self.reset()

            print("self.initial_ee_location: "
                  f"{self.initial_ee_location[0].numpy()}")
            print(f"self.target_ee_location: "
                  "{self.target_ee_location[0].numpy()}\n")

            initial_distance = \
                torch.cdist(torch.unsqueeze(self.ee_location,        dim=0),
                            torch.unsqueeze(self.target_ee_location, dim=0)
                            ).item()

            print(f"initial_distance: {initial_distance}\n")

            optimizer = torch.optim.Adam(self.controller.parameters(),
                                         lr=self.learning_rate)

            # reset and set optimizer over trials,
            # calculate and plot reward and mean error over max_iterations

            optimInfo = {"loss": [], "time": []}

            self.controller.train()

            start_model_training = time.perf_counter()

            for i in range(self.iterations):

                optimizer.zero_grad()

                print(f"trial {trial + 1}/{self.trials}, "
                      f"iteration {i + 1}/{self.iterations}")

                reward = self.calculate_step_reward()

                loss = -reward

                loss.backward()

                distance = \
                    torch.cdist(
                        torch.unsqueeze(self.ee_location,        dim=0),
                        torch.unsqueeze(self.target_ee_location, dim=0)
                        ).to('cpu').detach().item()

                percent_distance_covered = (  (initial_distance - distance)
                                            / initial_distance)

                print("percent distance covered: "
                      f"{percent_distance_covered:.1%}\n")

                optimizer.step()

                optimInfo["loss"].append(loss.item())

            # TODO some modifications that plots are correct

            # plot_policy()

            print(f"trial {trial + 1} distance covered: "
                  f"{percent_distance_covered:.1%}\n")

            if False:
                _, ax_loss = plt.subplots(figsize=(6, 4))

                ax_loss.plot(optimInfo["loss"], label='Training Loss')

                ax_loss.set_title('Training Loss Over Iterations')
                ax_loss.set_xlabel('Iteration')
                ax_loss.set_ylabel('Loss')

                ax_loss.legend()

                plt.show()

            torch.save(self.controller.state_dict(),
                       f"trained_models/RL-{trial + 1:03}.pth")

        # TODO: better printing function
        # print("Optimized data: ", all_optim_data)

        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("Training time: ", elapsed_model_training)

    def calculate_step_reward(self):
        """calculate and return reward for a single time step"""

        def print_value(title, tensor):
            """print a tensor's value for debugging"""

            print(f"{title}: {tensor.to('cpu').detach().numpy()}")

        self.joint_state = self.joint_state.to(self.device,
                                               dtype=self.dtype).detach()

        print(f"joint state: {self.joint_state.cpu().numpy()}")

        controller_inputs = [self.state[features[feature]]
                             for feature in
                             ("joints", "velocities", "accelerations")]

        controller_inputs = torch.unsqueeze(torch.cat(inputs), dim=0)

        action = self.controller(controller_inputs)

        print_value("action", action)

        gp_inputs = [self.state[features[feature]]
                     for feature in X_features]

        gp_inputs = torch.unsqueeze(torch.cat(gp_inputs), dim=0)

        print_value("gp_inputs", gp_inputs[0])

        predictions = self.gp_model.predict(inputs)

        for feature in y_features:

            (start_index,
             end_index) = get_feature_indices(feature_names = y_features,
                                              query_feature = feature)

            self.state[feature] = \
                torch.tensor(predictions.mean[0, start_index : end_index])

            self.state[feature].to(self.device, dtype=torch.float64)

        print_value("joints",        self.state["joints"])
        print_value("velocity",      self.state["velocities"])
        print_value("accelerations", self.state["accelerations"])

        print_value("current EE location", self.state["ee_location"])
        print_value("   goal EE location", self.target_ee_location[0])

        # TODO calculate the vector from predicted EE location towards goal

        # vector from the predicted EE coordinates towards the goal
        ideal_vector_to_goal = (  self.target_ee_location
                                - self.state["ee_location"])

        # TODO calculate the dot product between this and acceleration

        error = {}

        for error_metric in ("acceleration", "velocity"):

            dot_product = torch.dot(ideal_vector_to_goal,
                                    self.state[error_metric])

            error = (torch.arccos(dot_product)
                     / (  torch.norm(ideal_vector_to_goal)
                        * torch.norm(self.state[error_metric])))

        # TODO first try only acceleration error as loss
        # TODO then try combining with velocity error
        # TODO finally, try combining each/both with Euclidian distance

        euclidian_distance = \
            torch.cdist(torch.unsqueeze(self.state["ee_location"], dim=0),
                        torch.unsqueeze(self.target_ee_location,   dim=0))

        reward = -euclidian_distance

        print(f"reward: {reward.item()}")

        return euclidian_distance

    def reset(self):
        """set initial & goal states"""

        random_state = get_random_state(self.data_path)

        initial_state = {}

        for feature in X_features:

            data = torch.tensor([random_state[column]
                                 for column in features[feature]],
                                 self.dtype)

            initial_state[feature] = self.scalers[feature].transform(data)

            # previously we scaled:
            # initial_joint_state.view(1, -1).numpy())[0, :]

            for data in initial_state[feature]:
                data = torch.tensor(data)

        self.target_ee_location = deepcopy(initial_state["ee_location"])

        while (    (   self.target_ee_location[0, 0]
                    == initial_state["ee_location"][0, 0])
               and (   self.target_ee_location[0, 1]
                    == initial_state["ee_location"][0, 1])):

            print("BEFORE:")
            print("  initial_state['ee_location'][0, 0]: "
                  f"{initial_state['ee_location'][0, 0]}")
            print("  initial_state['ee_location'][0, 1]: "
                  f"{initial_state['ee_location'][0, 1]}")
            print("  target_ee_location[0, 0]: "
                  f"{self.target_ee_location[0, 0]}")
            print("  target_ee_location[0, 1]: "
                  f"{self.target_ee_location[0, 1]}\n")

            random_state = get_random_state(self.data_path)

            self.target_ee_location = \
                torch.tensor([
                  [random_state["boom_x"],
                   random_state["boom_y"]]
                  ], dtype=self.dtype)

            print("AFTER:")
            print("  initial_ee_location[0, 0]: "
                  f"{self.initial_ee_location[0, 0]}")
            print("  initial_ee_location[0, 1]: "
                  f"{self.initial_ee_location[0, 1]}")
            print("  target_ee_location[0, 0]: "
                  f"{self.target_ee_location[0, 0]}")
            print("  target_ee_location[0, 1]: "
                  f"{self.target_ee_location[0, 1]}\n")

        self.target_ee_location = \
            self.ee_location.transform(self.target_ee_location)

        self.target_ee_location = torch.tensor(self.target_ee_location)

        self.state = initial_state
