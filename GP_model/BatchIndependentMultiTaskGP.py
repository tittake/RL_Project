import gpytorch
import torch


class BatchIndependentMultiTaskGPModel(gpytorch.models.ExactGP):

    def __init__(self,
                 train_inputs,
                 train_targets,
                 likelihood,
                 num_tasks,
                 ard_num_dims):

        super().__init__(train_inputs  = train_inputs,
                         train_targets = train_targets,
                         likelihood    = likelihood)

        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks]))

        # ard stands for "Automatic Relevance Determination"
        self.cov_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape  = torch.Size([num_tasks]),
                ard_num_dims = ard_num_dims),
            batch_shape = torch.Size([num_tasks]))

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.cov_module(x)

        return \
            gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                )
