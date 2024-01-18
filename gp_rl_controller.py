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

    training_data_path = configuration["training_data_path"]

    # instantiate a new model for training
    gp_model = GPModel(data_path = training_data_path)

    # train and save weights
    gp_model.train(iterations    = configuration["training_iterations"],
                   save_model_to = configuration["model_path"],
                   plot_loss     = True)

    gp_model.test(plot=True)

    # reinstantiate the model from the saved weights
    gp_model = GPModel(saved_model_path = configuration["model_path"])

    # continue training on some other data, don't save changed model this time
    gp_model.train(iterations = configuration["training_iterations"],
                   data_path  = configuration["testing_data_path"],
                   plot_loss  = True)

if __name__ == "__main__":

    # TODO: Separate GP and RL configs

    with open("configuration.yaml") as configuration_file:

        configuration = yaml.load(configuration_file,
                                  Loader = yaml.Loader)

    main(configuration)
