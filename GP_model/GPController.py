from os import getcwd, mkdir
from os.path import dirname, exists, isdir, isfile, join
from pathlib import Path
import time

from joblib import dump, load
import gpytorch
import matplotlib.pyplot as plt
import torch

import data.dataloader as dataloader

from GP_model.BatchIndependentMultiTaskGP \
    import BatchIndependentMultiTaskGPModel


class GPModel:

    def __init__(self,
                 data_path:        str = None,
                 saved_model_path: str = None):
        '''
        Initializes Gaussian Process model. 
        '''

        self.metadata_attributes = ("input_feature_count",
                                    "output_feature_count",
                                    "joint_scaler",
                                    "torque_scaler",
                                    "ee_location_scaler")

        new_model_arguments = (data_path, )

        saved_model_arguments = (saved_model_path, )

        model_is_new = (    all(option is not None for option
                                in new_model_arguments)
                        and all(option is None for option
                                in saved_model_arguments))

        model_is_saved = (    all(option is not None
                                  for option in saved_model_arguments)
                          and all(option is None
                                  for option in new_model_arguments))

        try:
            assert (   (model_is_new   and not model_is_saved)
                    or (model_is_saved and not model_is_new))

        except AssertionError:
            raise ValueError("Please provide either:\n"
                             "  data_path; or"
                             "  saved_model_path, input_feature_count, "
                             "and output_feature_count.")

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        print(f"using device: {self.device}")

        if model_is_new:

            if isfile(data_path):
                (self.X_train,
                 self.y_train,
                 self.joint_scaler,
                 self.torque_scaler,
                 self.ee_location_scaler) = \
                    dataloader.load_training_data(
                        data_path = data_path,
                        normalize  = True)

            elif isdir(data_path):
                (self.X_train,
                 self.X_test,
                 self.y_train,
                 self.y_test,
                 self.joint_scaler,
                 self.torque_scaler,
                 self.ee_location_scaler) = \
                    dataloader.load_data_directory(data_path)

            else:
                raise ValueError("invalid path: " + data_path)

            self.X_train = self.X_train.to(self.device, dtype=torch.float64)
            self.y_train = self.y_train.to(self.device, dtype=torch.float64)

            self.input_feature_count = self.X_train.shape[1]

            self.output_feature_count = self.y_train.shape[1]

        if model_is_saved:

            self.X_train = None
            self.y_train = None

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

        self.likelihood = \
            gpytorch.likelihoods\
            .MultitaskGaussianLikelihood(num_tasks = self.output_feature_count,
                                         ).to(device = self.device,
                                              dtype  = torch.float64)

        self.model = \
            BatchIndependentMultiTaskGPModel(
                    train_inputs  = self.X_train,
                    train_targets = self.y_train,
                    likelihood    = self.likelihood,
                    num_tasks     = self.output_feature_count,
                    ard_num_dims  = self.input_feature_count,
                    ).to(self.device, torch.float64)

        if model_is_saved:

            state_dict = torch.load(saved_model_path)

            self.model.load_state_dict(state_dict)

    def train(self,
              iterations,
              data_path      = None,
              save_model_to  = None,
              plot_loss      = False):
        '''
        Training loop for GP. 
        Args:
            iterations: (int) Training loop iterations
            data_path: Path to trajectory data
            save_model_to: Path to save trained model parameters to 
            plot_loss (bool): Plot loss function or not  
        '''

        if data_path is None:
            if self.X_train is None:
                raise TypeError("data_path must be provided to train!")

        else:

            # TODO pass existing scalers from pretrained model to dataloader!!!

            if isfile(data_path):

                # ignore returned scalers, keep scalers from pretrained model
                (self.X_train, self.y_train, _, _, _) = \
                    dataloader.load_training_data(
                        data_path = data_path,
                        normalize  = True)

            elif isdir(data_path):

                # ignore returned scalers and testing dataset
                (self.X_train, _, self.y_train, _, _, _, _) = \
                    dataloader.load_data_directory(data_path)

            else:
                raise ValueError("invalid path: " + data_path)

            self.X_train = self.X_train.to(self.device, dtype=torch.float64)
            self.y_train = self.y_train.to(self.device, dtype=torch.float64)

            assert self.input_feature_count == self.X_train.shape[1]

            assert self.output_feature_count == self.y_train.shape[1]

            self.model.set_train_data(inputs  = self.X_train,
                                      targets = self.y_train,
                                      strict  = False)

        self.model.train()

        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        loss_metric = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                               self.model)

        start_model_training = time.perf_counter()

        self.loss_history = []

        for i in range(iterations):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -loss_metric(output, self.y_train)
            loss.backward()
            print(f"iteration {i + 1} / {iterations} - Loss: {loss.item()}")
            optimizer.step()
            self.loss_history.append(loss.item())

        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("Training time: ", elapsed_model_training)

        self.model.eval()
        self.likelihood.eval()

        if save_model_to:

            folder = dirname(join(getcwd(), save_model_to))

            if not isdir(folder):
                mkdir(folder)

            torch.save(self.model.state_dict(), save_model_to)

            metadata = tuple(getattr(self, attribute)
                            for attribute in self.metadata_attributes)

            dump(value    = metadata,
                 filename = join(folder,
                                 Path(save_model_to).stem
                                 + ".joblib"))

        if plot_loss:

            # Plot for training loss
            _, ax_loss = plt.subplots(figsize=(6, 4))

            ax_loss.plot(self.loss_history, label='Training Loss')
            ax_loss.set_title('Training Loss Over Iterations')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()

            plt.show()

    def test(self, data_path=None, plot=False):
        '''
        Function to test out trained Gaussian Process model. Plots GP predictions results.
        Args:
            data_path: Path for testing data
            plot (bool): Plot predicted means
        '''

        if data_path is None:
            if not hasattr(self, "X_test"):
                raise TypeError("data_path must be provided to test!")

        else:

            # TODO pass existing scalers from pretrained model to dataloader!!!

            if isfile(data_path):

                # ignore returned scalers, keep scalers from training
                (self.X_test,
                 self.y_test,
                 _, _, _) = \
                    dataloader.load_testing_data(
                        data_path = data_path,
                        normalize  = True)

            elif isdir(data_path):

                # ignore returned scalers and training dataset
                (_, self.X_test, _, self.y_test, _, _, _) = \
                    dataloader.load_data_directory(data_path)

            else:
                raise ValueError("invalid path: " + data_path)

        self.X_test = self.X_test.to(self.device, dtype=torch.float64)
        self.y_test = self.y_test.to(self.device, dtype=torch.float64)

        # Plot for tasks
        tasks = ["End-effector x-location", "End-effector y-location"]

        self.model.eval()
        self.likelihood.eval()

        if plot:
            _, axes_tasks = plt.subplots(1, len(tasks), figsize=(12, 4))

        for i, task in enumerate(tasks):

            # Make predictions for each task
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, 1, len(self.X_test[:, 0]))
                predictions = self.likelihood(self.model(self.X_test))
                mean = predictions.mean
                lower, upper = predictions.confidence_region()

            if plot:

                # Plot training data as black stars
                axes_tasks[i].plot(test_x.cpu().numpy(),
                                   self.y_test[:, i].cpu().numpy(), 'k*')

                axes_tasks[i].plot(test_x.cpu().numpy(),
                                   mean[:, i].cpu().numpy(), 'b')

                # Shade in confidence
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
        '''
        Predict end-effector location and joint values for a single joint value+torque sample.
        Args:
            X: Input sample (3x joint values + 3x joint torques)
        Returns:
            observed_pred: Predicted next end-effector location and corresponding joint values
        '''
        with gpytorch.settings.fast_pred_var():  # torch.no_grad(),
            observed_pred = self.likelihood(self.model(X))
        return observed_pred
