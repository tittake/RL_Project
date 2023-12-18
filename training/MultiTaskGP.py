import torch
import gpytorch


class MultitaskGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood, num_tasks):
    super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.MultitaskMean(
      gpytorch.means.ConstantMean(), num_tasks
    )
    self.covar_module = gpytorch.kernels.MultitaskKernel(
      gpytorch.kernels.RBFKernel(), num_tasks, rank=2
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)