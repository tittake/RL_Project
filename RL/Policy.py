from copy import deepcopy
import time
import torch
import numpy as np

from RL.Controller import RlController
from RL.utils import get_random_state

from GP.GpModel import GpModel
import matplotlib.pyplot as plt


class PolicyNetwork:

    def __init__(self,
                 gp_model: GpModel,
                 data_path:            str,
                 state_feature_count:  int = 3,
                 control_output_count: int = 3,
                 trials:               int = 100,
                 iterations:           int = 1000,
                 learning_rate:      float = 0.1):

        super().__init__()

        self.gp_model      = gp_model
        self.data_path     = data_path
        self.iterations    = iterations
        self.trials        = trials
        self.learning_rate = learning_rate

        self.joint_scaler       = self.gp_model.joint_scaler
        self.torque_scaler      = self.gp_model.torque_scaler
        self.ee_location_scaler = self.gp_model.ee_location_scaler

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        self.dtype = torch.double

        self.controller = \
            RlController(state_feature_count  = state_feature_count,
                         control_output_count = control_output_count)

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

            initial_distance = \
                torch.cdist(torch.unsqueeze(self.ee_location,        dim=0),
                            torch.unsqueeze(self.target_ee_location, dim=0)
                            ).item()

            optimizer = torch.optim.Adam(self.controller.parameters(),
                                         lr=self.learning_rate)

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

            final_distance = \
                torch.cdist(torch.unsqueeze(self.ee_location,        dim=0),
                            torch.unsqueeze(self.target_ee_location, dim=0)
                            ).item()

            percent_distance_covered = (  (initial_distance - final_distance)
                                        / initial_distance)

            print(f"trial {trial + 1} distance covered: "
                  f"{percent_distance_covered:.1%}\n")

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

        # plot losses from all iterations for each trial at the end

        if isinstance(axs, np.ndarray): # multiple subplots

            time_step_count = len(all_optim_data["all_optimizer_data"])

            fig, axs = \
                plt.subplots(len(all_optim_data["all_optimizer_data"]), 1,
                             figsize=(8, 6 * time_step_count))

            fig.suptitle('Loss Subplots', y=0.92)

            fig.tight_layout(pad=3.0)

            for idx, trial_save_info in \
                    enumerate(all_optim_data["all_optimizer_data"]):

                ax = axs[idx]
                ax.set_title(f'Trial {trial_save_info["trial"]}')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                losses = trial_save_info["optimInfo"]["loss"]
                ax.plot(range(1, len(losses) + 1), losses)

        else: # single subplot

            fig, ax = plt.subplots(figsize=(8, 6))

            fig.suptitle('Loss Subplots', y=0.92)

            title = f'Trial {all_optim_data["all_optimizer_data"][0]["trial"]}'

            ax.set_title(title)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')

            losses = (all_optim_data["all_optimizer_data"][0]
                      ["optimInfo"]["loss"])

            ax.plot(range(1, len(losses) + 1), losses)

        plt.show()

    def calculate_step_reward(self):
        """calculate and return reward for a single time step"""

        def print_value(title, tensor):
            """print a tensor's value for debugging"""

            print(f"{title}: {tensor.to('cpu').detach().numpy()}")

        self.joint_state = self.joint_state.to(self.device,
                                               dtype=self.dtype).detach()

        print(f"joint state: {self.joint_state.cpu().numpy()}")

        action = self.controller(self.joint_state)

        print_value("action", action)

        inputs = torch.unsqueeze(torch.cat([self.joint_state, action]), dim=0)

        print_value("inputs", inputs[0])

        # predict next state

        predictions = self.gp_model.predict(inputs)

        self.ee_location = predictions.mean[0, 0:2]
        self.joint_state = predictions.mean[0, 2:]

        print_value("joint state", self.joint_state)

        print_value("current EE location", self.ee_location)
        print_value("   goal EE location", self.target_ee_location[0])

        self.ee_location = self.ee_location.to(self.device,
                                               dtype=torch.float64)

        self.target_ee_location = \
            self.target_ee_location.to(self.device, dtype=torch.float64)

        reward = \
            -torch.cdist(torch.unsqueeze(self.ee_location,        dim=0),
                         torch.unsqueeze(self.target_ee_location, dim=0))

        print(f"reward: {reward.item()}")

        return reward

    def reset(self):
        """set initial & goal states"""

        random_state = get_random_state(self.data_path)

        self.initial_joint_state = \
            torch.tensor([random_state["theta1"],
                          random_state["theta2"],
                          random_state["xt2"]],
                         dtype=self.dtype)

        self.initial_joint_state = \
            self.joint_scaler\
                .transform(self.initial_joint_state.view(1, -1).numpy())[0, :]

        self.initial_ee_location = \
            torch.tensor([
              [random_state["boom_x"],
               random_state["boom_y"]]
              ], dtype=self.dtype)

        self.initial_ee_location = \
            self.ee_location_scaler.transform(self.initial_ee_location)

        random_state = get_random_state(self.data_path)

        self.target_ee_location = \
            torch.tensor([
              [random_state["boom_x"],
               random_state["boom_y"]]
              ], dtype=self.dtype)

        self.target_ee_location = \
            self.ee_location_scaler.transform(self.target_ee_location)

        self.initial_joint_state = torch.tensor(self.initial_joint_state)
        self.initial_ee_location = torch.tensor(self.initial_ee_location)
        self.target_ee_location = torch.tensor(self.target_ee_location)

        self.joint_state = deepcopy(self.initial_joint_state)
        self.ee_location = deepcopy(self.initial_ee_location)
