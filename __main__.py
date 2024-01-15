from argparse import ArgumentParser

def main():

    arguments_parser = ArgumentParser()

    subparsers = arguments_parser.add_subparsers()

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

    GP_model_class_choices = ("BatchIndependentMultiTaskGP",
                              "ExactGP",
                              "GPController",
                              "MultiTaskGP")

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

def train_GP(data_path, iterations, model_path, model_class):
    pass # TODO

def test_GP(data_path, plot):
    pass # TODO

if __name__ == "__main__":
    main()
