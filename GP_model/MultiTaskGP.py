import gpytorch


class MultitaskGPModel(gpytorch.models.ExactGP):

    def __init__(self, likelihood, num_tasks):

        super().__init__(train_inputs  = None,
                         train_targets = None,
                         likelihood    = likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks)

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks, rank=2)

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,
                                                                  covar_x)
