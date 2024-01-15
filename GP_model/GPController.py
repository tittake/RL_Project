import time
from os.path import isdir, isfile

import gpytorch
import matplotlib.pyplot as plt
import torch

import data.dataloader as dataloader

from GP_model.BatchIndependentMultiTaskGP \
    import BatchIndependentMultiTaskGPModel


class GPModel:

    def __init__(self, training_data_path):

        # TODO what are my arguments?

        super(GPModel, self).__init__()

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        print(f"using device: {self.device}")

        if isfile(training_data_path):
            self.X_train, self.y_train = \
                dataloader.load_training_data(train_path = training_data_path,
                                              normalize  = True)

        elif isdir(training_data_path):
            self.X_train, self.X_test, self.y_train, self.y_test = \
                dataloader.load_data_directory(training_data_path)

        input_feature_count = self.X_train.shape[1]

        output_feature_count = self.y_train.shape[1]

        self.likelihood = \
            gpytorch.likelihoods\
            .MultitaskGaussianLikelihood(num_tasks = output_feature_count,
                                         ).to(device = self.device,
                                              dtype  = torch.float64)

        self.model = \
            BatchIndependentMultiTaskGPModel(
                    likelihood   = self.likelihood,
                    num_tasks    = output_feature_count,
                    ard_num_dims = input_feature_count,
                    ).to(self.device, torch.float64)

    def load_saved_model(self):

        state_dict = torch.load(self.model_path)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.likelihood.eval()

    def train(self, iterations, save_model_to=None, plot_loss=False):

        self.X_train = self.X_train.to(self.device, dtype=torch.float64)
        self.y_train = self.y_train.to(self.device, dtype=torch.float64)

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
            torch.save(self.model.state_dict(), model_path)

        if plot_loss:

            # Plot for training loss
            _, ax_loss = plt.subplots(figsize=(6, 4))

            ax_loss.plot(self.loss_history, label='Training Loss')
            ax_loss.set_title('Training Loss Over Iterations')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()

            plt.show()

    def test(self, data_path, plot=False):

        self.X_test, self.y_test = \
            dataloader.load_test_data(test_path = data_path,
                                      normalize = True)

        self.X_test = self.X_test.to(self.device, dtype=torch.float64)
        self.y_test = self.y_test.to(self.device, dtype=torch.float64)

        # Plot for tasks
        tasks = ["x_boom", "y_boom"]

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
        with gpytorch.settings.fast_pred_var():  # torch.no_grad(),
            observed_pred = self.likelihood(self.model(X))
        return observed_pred
