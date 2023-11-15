import matplotlib.pyplot as plt
import gpytorch
import torch
import time
import data.dataloader as dataloader
from training.ExactGP import ExactGPModel
from training.BatchIndependentMultiTaskGP import BatchIndependentMultiTaskGPModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

torch.cuda.memory_summary(device=None, abbreviated=False)

class GPModel:
    def __init__(self):
        super(GPModel, self).__init__()
        
        # Check if a GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def initialize_model(self, train_path, test_path, num_tasks):
        # Load data
        self.num_tasks = num_tasks
        self.X_train, self.y_train = dataloader.load_training_data(train_path=train_path, normalize=True)
        self.X_test, self.y_test = dataloader.load_test_data(test_path=test_path, normalize=True)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(self.num_tasks).to(device=self.device, dtype=torch.float)
        self.model = BatchIndependentMultiTaskGPModel(self.X_train, self.y_train, self.likelihood, self.num_tasks).to(self.device, torch.float)

        # Move data tensors to the GPU
        self.X_train = self.X_train.to(self.device, dtype=torch.float)
        self.y_train = self.y_train.to(self.device, dtype=torch.float)
        self.X_test = self.X_test.to(self.device, dtype=torch.float)
        self.y_test = self.y_test.to(self.device, dtype=torch.float)

    def train(self, training_iter):

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # Loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        start_model_training = time.perf_counter()
        
        #scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.y_train)#.sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()

        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("Training time: ", elapsed_model_training)

        self.model.eval()
        self.likelihood.eval()

    def plot_training_results(self):
                
        # Initialize plot
        fig, axes = plt.subplots(1, self.num_tasks, figsize=(15, 5))

        # Define the tasks
        #tasks = ["theta1", "theta2", "xt2"]
        
        tasks = ["theta1"]
        
        # Make predictions for the single task
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 10, len(self.X_test[:, 0]))
            predictions = self.likelihood(self.model(self.X_test))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        # Create a single plot for the task
        plt.plot(test_x.cpu().numpy(), self.y_train[:, 0].cpu().numpy(), 'k*')
        plt.plot(test_x.cpu().numpy(), mean[:, 0].cpu().numpy(), 'b')
        # Shade in confidence
        plt.fill_between(test_x.cpu().numpy(), lower[:, 0].cpu().numpy(), upper[:, 0].cpu().numpy(), alpha=0.5)
        #plt.ylim(-3, 3)
        plt.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.title(f'Observed Values (Likelihood) for theta1')

        plt.show()

        #Plot multiple results
        # for i, task in enumerate(tasks):
        #     # Make predictions for each task
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         test_x = torch.linspace(0, 1, len(X_test[:,0]))
        #         predictions = likelihood(model(X_test))
        #         mean = predictions.mean
        #         lower, upper = predictions.confidence_region()

        #     # Plot training data as black stars
        #     axes[i].plot(test_x.cpu().numpy(), y_train[:, i].cpu().numpy(), 'k*')
        #     axes[i].plot(test_x.cpu().numpy(), mean[:, i].cpu().numpy(), 'b')
        #     # Shade in confidence
        #     axes[i].fill_between(test_x.cpu().numpy(), lower[:, i].cpu().numpy(), upper[:, i].cpu().numpy(), alpha=0.5)
        #     axes[i].set_ylim([-3, 3])
        #     axes[i].legend(['Observed Data', 'Mean', 'Confidence'])
        #     axes[i].set_title('Observed Values (Likelihood)')

        # plt.show()

    def predict(self, X):
        with gpytorch.settings.fast_pred_var():  # torch.no_grad(),
            observed_pred = self.likelihood(self.model(X))
        return observed_pred