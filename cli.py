"""CLI for GP and RL models"""

from argparse import ArgumentParser

from GP.GpModel import GpModel
from RL.Policy import PolicyNetwork

# TODO change all imports to package-relative to enable running as module
# TODO test if argcomplete will work for tab completion on windows too


def add_GP_training_parser_arguments(parser):
    """`GP train` arguments"""

    parser.set_defaults(function=train_GP)

    parser.add_argument("--data_path",
                        type     = str,
                        required = True)

    parser.add_argument("--iterations",
                        type     = int,
                        required = False,
                        default  = 150)

    parser.add_argument("--save_model_to",
                        type     = str,
                        required = False)

    # TODO add support for swapping between modular model architectures

    # GP_model_class_choices = ("BatchIndependentMultiTaskGP", "MultiTaskGP")

    # parser.add_argument("--model_class",
                        # type    = str,
                        # choices = GP_model_class_choices,
                        # default = "MultiTaskGP")

    parser.add_argument("--plot_loss",
                        type     = bool,
                        required = False,
                        default  = False)

    parser.add_argument("--test",
                        type     = bool,
                        required = False,
                        default  = False)

def train_GP(data_path:     str,
             iterations:    int,
             save_model_to: str,
             # model_class: str, # TODO
             plot_loss:     bool,
             test:          bool):

    """`GP train` behavior"""

    if not save_model_to:
        print("WARNING: no path given for argument: `save_model_to` - "
              "trained model will not be saved!\n")

    gp_model = GpModel(data_path = data_path)

    gp_model.train(iterations    = iterations,
                   save_model_to = save_model_to,
                   plot_loss     = plot_loss)

    if test:
        gp_model.test(data_path = data_path,
                      plot      = True)

def add_GP_testing_parser_arguments(parser):
    """`GP test` arguments"""

    parser.set_defaults(function=test_GP)

    parser.add_argument("--model_path",
                        type     = str,
                        required = True)

    parser.add_argument("--data_path",
                        type     = str,
                        required = True)

    parser.add_argument("--plot",
                        type     = bool,
                        required = False,
                        default  = True)

def test_GP(model_path: str,
            data_path:  str,
            plot:       bool):

    """`GP test` behavior"""

    gp_model = GpModel(data_path        = data_path,
                       saved_model_path = model_path)

    # TODO add accuracy metrics to GpModel.test(), then add option not to plot
    gp_model.test(plot = True)

def add_RL_training_parser_arguments(parser):
    """`RL train` arguments"""

    parser.set_defaults(function=train_RL)

    parser.add_argument("--data_path",
                        type     = str,
                        required = True)

    parser.add_argument("--GP_model_path",
                        "--gp_model_path",
                        dest     = "GP_model_path",
                        type     = str,
                        required = True)

    parser.add_argument("--trials",
                        type     = int,
                        required = False,
                        default  = 100)

    parser.add_argument("--iterations",
                        type     = int,
                        required = False,
                        default  = 1000)

    parser.add_argument("--learning_rate"
                        "--lr",
                        dest     = "learning_rate",
                        type     = float,
                        required = False,
                        default  = 0.01)

def train_RL(data_path:              str,
             GP_model_path:          str,
             trials:                 int,
             iterations:             int,
             learning_rate:          float):

    """`RL train` behavior"""

    gp_model = GpModel(data_path        = data_path,
                       saved_model_path = GP_model_path)

    policy_network = \
        PolicyNetwork(gp_model      = gp_model,
                      trials        = trials,
                      iterations    = iterations,
                      learning_rate = learning_rate)

    policy_network.optimize_policy()

    # TODO what if user wants to test the model just trained without saving it?


def main():
    """configure and run CLI parser"""

    arguments_parser = ArgumentParser()

    subparsers = arguments_parser.add_subparsers(required=True)

    GP_parser = subparsers.add_parser("GP", aliases=["gp"])
    RL_parser = subparsers.add_parser("RL", aliases=["rl"])

    GP_subparsers = GP_parser.add_subparsers(required=True)
    RL_subparsers = RL_parser.add_subparsers(required=True)

    GP_training_parser = GP_subparsers.add_parser("train")
    GP_testing_parser  = GP_subparsers.add_parser("test")
    RL_training_parser = RL_subparsers.add_parser("train")

    add_GP_training_parser_arguments(GP_training_parser)

    add_GP_testing_parser_arguments(GP_testing_parser)

    add_RL_training_parser_arguments(RL_training_parser)

    arguments = arguments_parser.parse_args()

    function = arguments.function

    arguments = {key: value for key, value
                 in vars(arguments).items()
                 if key != "function"}

    function(**arguments)


if __name__ == "__main__":
    main()
