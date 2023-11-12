from os.path import isfile
from sys import argv

from matplotlib import pyplot
from pandas import read_csv
import torch
from torch import tensor
from tqdm import tqdm, trange


MAX_DATAPOINTS = 800

TRAINING_ITERATIONS = 250

only_inputs = ["fc1", "fc2", "fct2"]

both_inputs_and_outputs = ["boom_x",
                           "boom_y",
                           "boom_z",
                           "boom_angle",
                           "boom_x_velocity",
                           "boom_y_velocity",
                           "boom_z_velocity",
                           "boom_x_acceleration",
                           "boom_y_acceleration",
                           "boom_z_acceleration",
                           "theta1",
                           "theta2"]


def load_data(path):

  trajectory = read_csv(path).iloc[:MAX_DATAPOINTS]

  inputs  = tensor(trajectory[both_inputs_and_outputs + only_inputs]
                   .iloc[:-1].values,
                   dtype=torch.float32)

  outputs = tensor((  trajectory[both_inputs_and_outputs]
                    - trajectory[both_inputs_and_outputs].shift(1)
                    ).iloc[1:].values,
                   dtype=torch.float32)

  # print(inputs.shape) # 10045, 15

  # print(outputs.shape) # 10045, 12

  return inputs, outputs


assert len(argv) == 3

for path in argv[1:3]:
  assert isfile(path)


import math
import torch
import gpytorch

class MultitaskGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.MultitaskMean(
      gpytorch.means.ConstantMean(), num_tasks=12
    )
    self.covar_module = gpytorch.kernels.MultitaskKernel(
      gpytorch.kernels.RBFKernel(), num_tasks=12, rank=1
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

inputs, outputs = load_data(path=argv[1])

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=12)
model = MultitaskGPModel(inputs, outputs, likelihood)


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# TODO try training on not only one trajectory?
  # iterate through trajectories, and also through "batches" of ~800 steps

for i in trange(TRAINING_ITERATIONS):
  optimizer.zero_grad()
  predictions = model(inputs)
  loss = -mll(predictions, outputs)
  loss.backward()
  print(f"iteration {i + 1} / {TRAINING_ITERATIONS}, loss: {loss.item()}")
  optimizer.step()


inputs, outputs = load_data(path=argv[2])

timestamps = read_csv(path).iloc[: MAX_DATAPOINTS - 1]["time"]

model.eval()
likelihood.eval()

axes = []

figure, axes = pyplot.subplots(1, outputs.shape[1], figsize=(8, 3))

with torch.no_grad(), gpytorch.settings.fast_pred_var():
  predictions = likelihood(model(inputs))
  mean = predictions.mean
  lower, upper = predictions.confidence_region()

for task in range(outputs.shape[1]):
  axes[task].plot(timestamps, outputs[:, task].detach().numpy(), "k*")
  axes[task].plot(timestamps, mean[:, task].numpy(), 'b')
  axes[task].fill_between(timestamps,
                     lower[:, task].numpy(),
                     upper[:, task].numpy(),
                     alpha=0.5)
  axes[task].set_ylim([-3, 3])
  axes[task].legend(["Observed Data", "Mean", "Confidence"])
  axes[task].set_title(both_inputs_and_outputs[task])

pyplot.show()
