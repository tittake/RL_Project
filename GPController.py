import matplotlib.pyplot as plt
import numpy as np
import gpytorch
import torch
import time

import data.dataloader as dataloader
from training.ExactGP import ExactGPModel
from training.BatchIndependentMultiTaskGP import BatchIndependentMultitaskGPModel
import matplotlib.pyplot as plt


train_path = "data/training1.csv"
test_path = "data/testing1.csv"
training_iter = 20 
num_tasks = 5

def main():
    #Load data

    X_train, y_train = dataloader.load_training_data(train_path=train_path, normalize=True)
    X_test, y_test = dataloader.load_test_data(test_path=test_path, normalize=True)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
    model = BatchIndependentMultitaskGPModel(X_train, y_train, likelihood, num_tasks)

    # Check if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Move data tensors to the GPU
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test = X_test.to(device)

    start_model_training = time.perf_counter()
    #Find optimal model hyperparameters
    model.train()
    likelihood.train()

    #Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    batch_size = 32  # Choose a smaller batch size

    #Loss
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    scaler = torch.cuda.amp.grad_scaler
    for i in range(training_iter):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast_mode.autocast():
            output = model(X_train)
            loss = -mll(output, y_train)
        scaler.GradScaler.scale(loss).backward()
        scaler.GradScaler.step(optimizer)
        scaler.GradScaler.update()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        
    end_model_training = time.perf_counter()
    elapsed_model_training = end_model_training - start_model_training
    print(elapsed_model_training)

    model.eval()
    likelihood.eval()

    #Initialize plot
    fig, axes = plt.subplots(1, num_tasks, figsize=(15, 5))

    # Define the tasks
    tasks = ["theta1", "theta2", "xt2", "x_boom", "y_boom"]
    
    for i, task in enumerate(tasks):
        # Make predictions for each task
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 1, 10058)
            predictions = likelihood(model(X_test))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        #Plot training data as black stars
        axes[i].plot(test_x.numpy(), y_train[i,:].numpy(), 'k*')
        axes[i].plot(test_x.numpy(), mean[:,i].numpy(), 'b')
        # Shade in confidence
        axes[i].fill_between(test_x.numpy(), lower[:, i].numpy(), upper[:, i].numpy(), alpha=0.5)
        axes[i].set_ylim([-1, 1])
        axes[i].legend(['Observed Data', 'Mean', 'Confidence'])
        axes[i].set_title('Observed Values (Likelihood)')
        
    plt.show()


if __name__ == "__main__":
    main() 