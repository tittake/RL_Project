import yaml

from GP_model.GPController import GPModel
from RL.policy import PolicyNetwork

#Initial RL testing values
#x_boom = 2.1 y_boom = 3.5
#x_boom = 3.0 y_boom = 2.4


def get_configs():
    return

def main(configuration):
    """Collective controller for GP model and RL controller"""

    #Initialize and train model or load pre-trained model
    gpmodel = GPModel(**configuration)

    gpmodel.plot_training_results()

    print(gpmodel.predict([0, 0, 0, 1, 1, 1]))

    policy_network = PolicyNetwork(**configuration)

    # policy_network.optimize_policy()

if __name__ == "__main__":

    #TODO: Separate GP and RL configs

    with open("configuration.yaml") as configuration_file:

        configuration = yaml.load(configuration_file,
                                  Loader = yaml.Loader)

    main(configuration)
