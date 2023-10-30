import matplotlib.pyplot as plt
import numpy as np
import gpytorch
import torch

import data.dataloader as dataloader
from training.ExactGP import ExactGPModel
from training.BatchIndependentMultiTaskGP import BatchIndependentMultitaskGPModel
import matplotlib.pyplot as plt


train_path = "data/training1.csv"
test_path = "data/testing1.csv"
training_iter = 1
num_tasks = 5

def main():
    #Load data
    X_train, y_train = dataloader.load_training_data(train_path=train_path, normalize=False)
    X_test, y_test = dataloader.load_test_data(test_path=test_path, normalize=False)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
    model = BatchIndependentMultitaskGPModel(X_train, y_train, likelihood, num_tasks)

    #Find optimal model hyperparameters
    model.train()
    likelihood.train()

    #Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    #Loss
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
    
    model.eval()
    likelihood.eval()

    #Initialize plot
    fig, axes = plt.subplots(1, num_tasks, figsize=(15, 5))

    # Define the tasks
    tasks = ["theta1", "theta2", "xt2", "x_boom", "y_boom"]
            
    # Make predictions for each task
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 10058)
        predictions = likelihood(model(X_test))
        mean = predictions.mean
        print(mean)
        lower, upper = predictions.confidence_region()
    print(test_x.shape)
    print(y_train[0,:].shape)
    #Plot training data as black stars
    axes[0].plot(test_x.numpy(), y_test[0,:].numpy(), 'k*')
    axes[0].plot(test_x.numpy(), mean[:,0].numpy(), 'b')
    # Shade in confidence
    axes[0].fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    axes[0].set_ylim([-3, 3])
    axes[0].legend(['Observed Data', 'Mean', 'Confidence'])
    axes[0].set_title('Observed Values (Likelihood)')
    fig.show()



if __name__ == "__main__":
    main()