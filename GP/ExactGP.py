import gpytorch


class ExactGpModel(gpytorch.models.ExactGP):

    def __init__(self, train_inputs, train_targets, likelihood):

        super().__init__(train_inputs  = train_inputs,
                         train_targets = train_inputs,
                         likelihood    = likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = \
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
