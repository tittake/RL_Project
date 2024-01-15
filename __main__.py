from argparse import ArgumentParser

# TODO migrate to package relative imports to be able to run as a module
from GP_model.GPController import GPModel

def main():

    arguments_parser = ArgumentParser()

    subparsers = arguments_parser.add_subparsers(required=True)

    GP_parser = subparsers.add_parser("GP")

    GP_subparsers = GP_parser.add_subparsers()

    GP_training_parser = GP_subparsers.add_parser("train")
    GP_testing_parser  = GP_subparsers.add_parser("test")

    GP_training_parser.set_defaults(function=train_GP)

    GP_training_parser.add_argument("--data_path",
                                    type     = str,
                                    required = True)

    GP_training_parser.add_argument("--iterations",
                                    type     = int,
                                    required = True)

    GP_training_parser.add_argument("--model_path",
                                    type = str)

    GP_model_class_choices = ("BatchIndependentMultiTaskGP", "MultiTaskGP")

    GP_training_parser.add_argument("--model_class",
                                    type    = str,
                                    choices = GP_model_class_choices,
                                    default = "MultiTaskGP")

    GP_testing_parser.set_defaults(function=test_GP)

    GP_testing_parser.add_argument("--data_path",
                                   type     = str,
                                   required = True)

    GP_testing_parser.add_argument("--plot",
                                   type = bool)

    arguments = arguments_parser.parse_args()

    function = arguments.function

    arguments = {key: value for key, value
                 in vars(arguments).items()
                 if key != "function"}

    function(**arguments)

def train_GP(data_path:   str,
             iterations:  int,
             model_path:  str,
             model_class: str,
             plot_loss:   bool):

    gp_model = GPModel(training_data_path = data_path)

    gp_model.train(iterations=iterations, plot_loss=plot_loss)

    # TODO what if user wants to test the model just trained without saving it?

def test_GP(model_path: str,
            data_path:  str,
            plot:       bool):

    pass # TODO instantiate model
    # is this even possible for a saved model without providing training data?

if __name__ == "__main__":
    main()
