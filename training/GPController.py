import matplotlib.pyplot as plt
import gpytorch
import torch
import time
import data.dataloader as dataloader
from training.ExactGP import ExactGPModel
from training.BatchIndependentMultiTaskGP import BatchIndependentMultiTaskGPModel
from training.MultiTaskGP import MultitaskGPModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

#torch.cuda.memory_summary(device=None, abbreviated=False)

class GPModel:
    def __init__(self, **params):
        super(GPModel, self).__init__()

        for key, value in params.items():
            setattr(self, key, value)
        
        # Check if a GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.initialize_model()


    def initialize_model(self):
        # Load data
        #self.num_tasks = num_tasks
        #self.ard_num_dims = ard_num_dims
        self.X_train, self.y_train = dataloader.load_training_data(train_path=self.train_path, normalize=True)
        self.X_test, self.y_test = dataloader.load_test_data(test_path=self.test_path, normalize=True)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks).to(device=self.device, dtype=torch.float64)
        
        self.model = BatchIndependentMultiTaskGPModel(self.X_train, self.y_train, self.likelihood, self.num_tasks, self.ard_num_dims).to(self.device, torch.float64)
        #self.model = MultitaskGPModel(self.X_train, self.y_train, self.likelihood, self.num_tasks).to(self.device, torch.float64)

        # Move data tensors to the GPU
        self.X_train = self.X_train.to(self.device, dtype=torch.float64)
        self.y_train = self.y_train.to(self.device, dtype=torch.float64)
        self.X_test = self.X_test.to(self.device, dtype=torch.float64)
        self.y_test = self.y_test.to(self.device, dtype=torch.float64)

        if self.train_GP:
            self.train()
        else:
            self.load_model()


    def train(self):
        
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
   
        # Loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        start_model_training = time.perf_counter()
        
        #scaler = torch.cuda.amp.grad_scaler.GradScaler()
        self.loss_history = []
        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iter, loss.item()))
            optimizer.step()
            self.loss_history.append(loss.item())


        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        print("Training time: ", elapsed_model_training)

        self.model.eval()
        self.likelihood.eval()

        #Save trained model
        torch.save(self.model.state_dict(), 'trained_models/two_joints_GP.pth')
        

    def plot_training_results(self):
                
        # Initialize plot
        fig, axes = plt.subplots(1, self.num_tasks+1, figsize=(15, 5))

        if(not self.train_GP):
            self.loss_history = []
        axes[0].plot(self.loss_history, label='Training Loss')
        axes[0].set_title('Training Loss Over Iterations')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        # Make predictions for the single task
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
            
        #     print(self.X_test.shape)
        #     test_x = torch.linspace(0, 10 ,len(self.X_test[:, 0]))
        #     predictions = self.likelihood(self.model(self.X_test))
        #     mean = predictions.mean
        #     lower, upper = predictions.confidence_region()

        # Create a single plot for the task
        #plt.plot(test_x.cpu().numpy(), self.y_test[:, 0].cpu().numpy(), 'k*')
        #plt.plot(test_x.cpu().numpy(), self.X_train[:, 1].cpu().numpy(), 'r*')
        #plt.plot(test_x.cpu().numpy(), mean[:, 0].cpu().numpy(), 'b')
        # Shade in confidence
        #plt.fill_between(test_x.cpu().numpy(), lower[:, 0].cpu().numpy(), upper[:, 0].cpu().numpy(), alpha=0.5)
        #plt.ylim(-3, 3)
        #plt.legend(['Observed Data', 'Mean', 'Confidence'])
        #plt.title(f'Observed Values (Likelihood) for theta1')

        #plt.show()

        #Plot multiple results
        # Define the tasks
        tasks = ["x_boom", "y_boom"]

        for i, task in enumerate(tasks):
            # Make predictions for each task
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, 1, len(self.X_test[:,0]))
                predictions = self.likelihood(self.model(self.X_test))
                mean = predictions.mean
                lower, upper = predictions.confidence_region()

            # Plot training data as black stars
            axes[i+1].plot(test_x.cpu().numpy(), self.y_test[:, i].cpu().numpy(), 'k*')
            axes[i+1].plot(test_x.cpu().numpy(), mean[:, i].cpu().numpy(), 'b')
            # Shade in confidence
            axes[i+1].fill_between(test_x.cpu().numpy(), lower[:, i].cpu().numpy(), upper[:, i].cpu().numpy(), alpha=0.5)
            axes[i+1].set_ylim([-3, 3])
            axes[i+1].legend(['Observed Data', 'Mean', 'Confidence'])
            axes[i+1].set_title('Observed Values (Likelihood), ' + task)

        plt.show()

    def predict(self, X):
        with gpytorch.settings.fast_pred_var():  # torch.no_grad(),
            observed_pred = self.likelihood(self.model(X))
        return observed_pred
    
    def load_model(self):
        state_dict = torch.load(self.model_path)
        self.model = BatchIndependentMultiTaskGPModel(self.X_train, self.y_train, self.likelihood, self.num_tasks, self.ard_num_dims).to(self.device, torch.float64)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.likelihood.eval()
        