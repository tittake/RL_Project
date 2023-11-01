import matplotlib.pyplot as plt
import gpytorch
import torch
import time
import data.dataloader as dataloader
from training.ExactGP import ExactGPModel
from training.BatchIndependentMultiTaskGP import BatchIndependentMultitaskGPModel
from torch.utils.data import DataLoader, TensorDataset

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

train_path = "data/training1.csv"
test_path = "data/testing1.csv"
training_iter = 20
num_tasks = 3
batch_size = 16  # Updated batch size

def main():
    # Load data
    X_train, y_train = dataloader.load_training_data(train_path=train_path, normalize=True)
    X_test, y_test = dataloader.load_test_data(test_path=test_path, normalize=True)

    # Check if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
    model = BatchIndependentMultitaskGPModel(X_train, y_train, likelihood, num_tasks).to(device, torch.float)

    # Move data tensors to the GPU
    X_train = X_train.to(device, dtype=torch.float)
    y_train = y_train.to(device, dtype=torch.float)
    print(X_train.shape)
    print(y_train.shape)
    # Create a DataLoader with the specified batch size
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_test = X_test.to(device, dtype=torch.float)

    start_model_training = time.perf_counter()
   
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Loss
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    end_model_training = time.perf_counter()
    elapsed_model_training = end_model_training - start_model_training
    print(elapsed_model_training)

    model.eval()
    likelihood.eval()

    # Initialize plot
    fig, axes = plt.subplots(1, num_tasks, figsize=(15, 5))

    # Define the tasks
    tasks = ["theta1", "theta2", "xt2"]

    for i, task in enumerate(tasks):
        # Make predictions for each task
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 1, 9959)
            predictions = likelihood(model(X_test))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        # Plot training data as black stars
        axes[i].plot(test_x.cpu().numpy(), y_train[:, i].cpu().numpy(), 'k*')
        axes[i].plot(test_x.cpu().numpy(), mean[:, i].cpu().numpy(), 'b')
        # Shade in confidence
        axes[i].fill_between(test_x.cpu().numpy(), lower[:, i].cpu().numpy(), upper[:, i].cpu().numpy(), alpha=0.5)
        axes[i].set_ylim([-3, 3])
        axes[i].legend(['Observed Data', 'Mean', 'Confidence'])
        axes[i].set_title('Observed Values (Likelihood)')

    plt.show()

if __name__ == "__main__":
    main()