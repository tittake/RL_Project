import time

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import data.dataloader as dataloader
from GP_model.ExactGP import ExactGPModel
from GP_model.BatchIndependentMultiTaskGP import BatchIndependentMultiTaskGPModel
from GP_model.MultiTaskGP import MultitaskGPModel

class GPModel:

    def __init__(self, **params):
        super(GPModel, self).__init__()

        for key, value in params.items():
            setattr(self, key, value)

        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        print(self.device)

        self.initialize_model()


    def initialize_model(self):

        self.X_train, self.y_train = \
            dataloader.load_training_data(train_path = self.train_path,
                                          normalize  = True)

        self.X_test, self.y_test = \
            dataloader.load_test_data(test_path = self.test_path,
                                      normalize = True)

        #Use whole data directory instead
        self.X_train, self.X_test, self.y_train, self.y_test = \
            dataloader.load_data_directory(self.data_directory)

        self.likelihood = \
            gpytorch.likelihoods\
            .MultitaskGaussianLikelihood(num_tasks = self.num_tasks
                                         ).to(device = self.device,
                                              dtype  = torch.float64)

        self.model = \
            BatchIndependentMultiTaskGPModel(self.X_train,
                                             self.y_train,
                                             self.likelihood,
                                             self.num_tasks,
                                             self.ard_num_dims
                                             ).to(self.device, torch.float64)

        self.X_train = self.X_train.to(self.device, dtype=torch.float64)
        self.y_train = self.y_train.to(self.device, dtype=torch.float64)
        self.X_test = self.X_test.to(self.device, dtype=torch.float64)
        self.y_test = self.y_test.to(self.device, dtype=torch.float64)

        if self.train_GP:
            self.train()
        else:
            self.load_model()


    def train(self):

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        loss_metric = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                               self.model)

        start_model_training = time.perf_counter()

        self.loss_history = []

        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -loss_metric(output, self.y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f'
                  % (i + 1, self.training_iter, loss.item()))
            optimizer.step()
            self.loss_history.append(loss.item())


        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("Training time: ", elapsed_model_training)

        self.model.eval()
        self.likelihood.eval()

        #Save trained model
        # torch.save(self.model.state_dict(),
                   # 'trained_models/two_joints_GP.pth')


    def plot_training_results(self):
        # Plot for training loss
        fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
        if not self.train_GP:
            self.loss_history = []
        ax_loss.plot(self.loss_history, label='Training Loss')
        ax_loss.set_title('Training Loss Over Iterations')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        plt.show()

        # Plot for tasks
        tasks = ["x_boom", "y_boom"]
        fig_tasks, axes_tasks = plt.subplots(1, len(tasks), figsize=(12, 4))

        for i, task in enumerate(tasks):
            # Make predictions for each task
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, 1, len(self.X_test[:,0]))
                predictions = self.likelihood(self.model(self.X_test))
                mean = predictions.mean
                lower, upper = predictions.confidence_region()

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
            axes_tasks[i].legend(['Observed Data', 'Mean', 'Confidence'])
            axes_tasks[i].set_title('Observed Values (Likelihood), ' + task)

        plt.show()


    def predict(self, X):
        with gpytorch.settings.fast_pred_var():  # torch.no_grad(),
            observed_pred = self.likelihood(self.model(X))
        return observed_pred

    def load_model(self):

        state_dict = torch.load(self.model_path)

        self.model = BatchIndependentMultiTaskGPModel(self.X_train,
                                                      self.y_train,
                                                      self.likelihood,
                                                      self.num_tasks,
                                                      self.ard_num_dims
                                                      ).to(self.device,
                                                           torch.float64)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.likelihood.eval()
