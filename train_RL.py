"""train RL-based controller"""

import yaml

from GP.GpModel import GpModel
from RL.Policy import RlPolicy


def main(configuration):
    """train or load GP model, then use it to train an RL model"""

    if configuration["RL"]["train_fresh_GP"]:

        gp_model = GpModel(data_path = configuration["data_path"])

        gp_model.train(iterations    = configuration["GP"]["iterations"],
                       save_model_to = configuration["GP"]["model_path"],
                       plot_loss     = True)

    else:

        gp_model = \
            GpModel(data_path        = configuration["data_path"],
                    saved_model_path = configuration["GP"]["model_path"])

    gp_model.test(plot=True)

    rl_policy = RlPolicy(gp_model  = gp_model,
                         data_path = configuration["data_path"])

    rl_policy.train(iterations    = configuration["RL"]["iterations"],
                    learning_rate = configuration["RL"]["learning_rate"])


if __name__ == "__main__":

    with open("configuration.yaml") as configuration_file:

        configuration = yaml.load(configuration_file,
                                  Loader = yaml.SafeLoader)

    main(configuration)
