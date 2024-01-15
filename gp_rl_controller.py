import yaml

from GP_model.GPController import GPModel
# from RL.policy import PolicyNetwork

# Initial RL testing values
# x_boom = 2.1 y_boom = 3.5
# x_boom = 3.0 y_boom = 2.4


def get_configs():
    return


def main(configuration):
    """Collective controller for GP model and RL controller"""

    gp_model = GPModel(training_data_path = configuration["data_directory"])

    gp_model.train(iterations=configuration["training_iter"], plot_loss=True)

    gp_model.test(data_path = configuration["test_path"], plot=True)

    from torch import tensor
    print(gp_model.predict(tensor([[0, 0, 0, 1, 1, 1]])).mean)

    # policy_network = PolicyNetwork(**configuration)

    # policy_network.optimize_policy()


if __name__ == "__main__":

    # TODO: Separate GP and RL configs

    with open("configuration.yaml") as configuration_file:

        configuration = yaml.load(configuration_file,
                                  Loader = yaml.Loader)

    main(configuration)
