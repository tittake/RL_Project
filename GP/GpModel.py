"""module containing class to model a Gaussian Process"""

from math import inf
from os import getcwd, mkdir
from os.path import dirname, isdir, isfile, join
from pathlib import Path
import time

from joblib import dump, load
import gpytorch
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import data.dataloader as dataloader

from GP.BatchIndependentMultiTaskGP import BatchIndependentMultiTaskGpModel


class GpModel:
    """Gaussian Process model"""

    def __init__(self,
                 data_path:        str,
                 saved_model_path: str = None):
        """loads data and initializes model"""

        self.metadata_attributes = ("input_features",
                                    "output_features",
                                    "scalers")

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        print(f"using device: {self.device}")

        if isfile(data_path):

            self.X_train, self.y_train = \
                dataloader.load_training_data(data_path = data_path,
                                              normalize  = True)

        elif isdir(data_path):

            (self.X_train,
             self.X_test,
             self.y_train,
             self.y_test) = \
                dataloader.load_data_directory(data_path)

        else:
            raise ValueError("invalid path: " + data_path)

        self.scalers = dataloader.scalers

        self.X_train = self.X_train.to(self.device, dtype=torch.float64)
        self.y_train = self.y_train.to(self.device, dtype=torch.float64)

        if not saved_model_path: # new model

            self.input_features  = dataloader.X_names
            self.output_features = dataloader.y_names

            self.input_feature_count  = self.X_train.shape[1]
            self.output_feature_count = self.y_train.shape[1]

            # verify that data's input size matches dataloader's input size
            assert len(self.input_features ) == self.input_feature_count
            assert len(self.output_features) == self.output_feature_count

        else: # pretrained model

            folder = dirname(join(getcwd(), saved_model_path))

            metadata_path = join(folder,
                                 Path(saved_model_path).stem
                                 + ".joblib")

            try:
                metadata = load(filename = metadata_path)
            except FileNotFoundError:
                raise FileNotFoundError("Saved scalers not found!\n"
                                        "  Expected path: " + metadata_path)

            for attribute_name, attribute \
                    in zip(self.metadata_attributes, metadata):

                setattr(self, attribute_name, attribute)

            # verify pretrained model trained on features matching dataloader's
            assert self.input_features  == dataloader.X_names
            assert self.output_features == dataloader.y_names

        self.likelihood = \
            gpytorch.likelihoods\
            .MultitaskGaussianLikelihood(num_tasks = len(self.output_features),
                                         ).to(device = self.device,
                                              dtype  = torch.float64)

        self.model = \
            BatchIndependentMultiTaskGpModel(
                train_inputs  = self.X_train,
                train_targets = self.y_train,
                likelihood    = self.likelihood,
                num_tasks     = len(self.output_features),
                ard_num_dims  = len(self.input_features)
            ).to(self.device, torch.float64)

        if saved_model_path:

            state_dict = torch.load(saved_model_path)

            self.model.load_state_dict(state_dict)

    def train(self,
              iterations,
              data_path      = None,
              save_model_to  = None,
              plot_loss      = False):
        """
        training loop for GP

        args:
            iterations (int): training loop iterations
            data_path:        path to trajectory data
            save_model_to:    path to save trained model parameters to
            plot_loss (bool): whether to plot the loss function
        """

        if data_path is None:
            if self.X_train is None:
                raise TypeError("data_path must be provided to train!")

        else:

            # TODO pass existing scalers from pretrained model to dataloader!!!

            if isfile(data_path):

                self.X_train, self.y_train = \
                    dataloader.load_training_data(data_path = data_path,
                                                  normalize  = True)

            elif isdir(data_path):

                self.X_train, _, self.y_train, _ = \
                    dataloader.load_data_directory(data_path)

            else:
                raise ValueError("invalid path: " + data_path)

            self.X_train = self.X_train.to(self.device, dtype=torch.float64)
            self.y_train = self.y_train.to(self.device, dtype=torch.float64)

            # verify pretrained model trained on data of same size as data
            assert len(self.input_features ) == self.X_train.shape[1]
            assert len(self.output_features) == self.y_train.shape[1]

            self.model.set_train_data(inputs  = self.X_train,
                                      targets = self.y_train,
                                      strict  = False)

        self.model.train()

        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        loss_metric = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                               self.model)

        start_model_training = time.perf_counter()

        best_loss = inf

        self.loss_history = []

        with tqdm(total = iterations) as progress_bar:

            for _ in range(iterations):

                optimizer.zero_grad()

                output = self.model(self.X_train)

                loss = -loss_metric(output, self.y_train)

                if loss.item() < best_loss:

                    best_loss = loss.item()

                    if save_model_to:
                        self.save_model(path = save_model_to)

                    progress_bar.set_description(f"loss: {loss.item()}")

                else:
                    progress_bar.set_description(
                        f"current loss: {loss.item()}, "
                        f"best: {best_loss}")

                loss.backward()
                optimizer.step()

                progress_bar.update()

                self.loss_history.append(loss.item())

        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("Training time: ", elapsed_model_training)

        self.model.eval()
        self.likelihood.eval()

        if plot_loss:

            # Plot for training loss
            _, ax_loss = plt.subplots(figsize=(6, 4))

            ax_loss.plot(self.loss_history, label='Training Loss')
            ax_loss.set_title('Training Loss Over Iterations')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()

            plt.show()

    def save_model(self, path):

        folder = dirname(join(getcwd(), path))

        if not isdir(folder):
            mkdir(folder)

        torch.save(self.model.state_dict(), path)

        metadata = tuple(getattr(self, attribute)
                         for attribute in self.metadata_attributes)

        dump(value    = metadata,
             filename = join(folder,
                             Path(path).stem
                             + ".joblib"))

    def test(self, data_path=None, plot=False): # TODO document return types
        """
        evaluate a trained Gaussian Process model on testing data

        args:
            data_path:   path for testing data
            plot (bool): whether to plot predicted means
        """

        if data_path is None:
            if not hasattr(self, "X_test"):
                raise TypeError("data_path must be provided to test!")

        else:

            # TODO pass existing scalers from pretrained model to dataloader!!!

            if isfile(data_path):

                (self.X_test, self.y_test) = \
                    dataloader.load_testing_data(data_path = data_path,
                                                 normalize  = True)

            elif isdir(data_path):

                _, self.X_test, _, self.y_test = \
                    dataloader.load_data_directory(data_path)

            else:
                raise ValueError("invalid path: " + data_path)

        self.X_test = self.X_test.to(self.device, dtype=torch.float64)
        self.y_test = self.y_test.to(self.device, dtype=torch.float64)

        # plot for tasks
        tasks = ["end-effector x-location",
                 "end-effector y-location",
                 "theta1",
                 "theta2",
                 "xt2",
                 "boom_x_velocity",  
                 "boom_y_velocity", 
                 "boom_x_acceleration", 
                 "boom_y_acceleration"]

        self.model.eval()
        self.likelihood.eval()

        if plot:
            _, axes_tasks = plt.subplots(1, len(tasks), figsize=(12, 4))

        for i, task in enumerate(tasks):

            # make predictions for each task
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, 1, len(self.X_test[:, 0]))
                predictions = self.likelihood(self.model(self.X_test))
                mean = predictions.mean
                lower, upper = predictions.confidence_region()

            if plot:

                # plot training data as black stars
                axes_tasks[i].plot(test_x.cpu().numpy(),
                                   self.y_test[:, i].cpu().numpy(), 'k*')

                axes_tasks[i].plot(test_x.cpu().numpy(),
                                   mean[:, i].cpu().numpy(), 'b')

                # shade in confidence
                axes_tasks[i].fill_between(test_x.cpu().numpy(),
                                           lower[:, i].cpu().numpy(),
                                           upper[:, i].cpu().numpy(),
                                           alpha=0.5)

                axes_tasks[i].set_ylim([-0.2, 1.3])
                axes_tasks[i].legend(["Observed Data", "Mean", "Confidence"])
                axes_tasks[i].set_title("Observed Values (Likelihood), "
                                        f"{task}")

        plt.show()

    def predict(self, X):
        """
        predicts next end-effector location and next joint values,
        given current joint state and torques applied

        args:
            X: input sample (3x joint values + 3x joint torques)
        returns:
            predicted next end-effector location & corresponding joint values
        """

        self.model.eval()
        self.likelihood.eval()

        prediction = self.likelihood(self.model(X))

        return prediction
