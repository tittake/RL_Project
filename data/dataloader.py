from itertools import chain
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch

dtype = torch.double

# TODO move features to a configuration file?

features  = {"ee_location":   ("boom_x", "boom_y"),
             "joints":        ("theta1", "theta2", "xt2"),
             "torques":       ("fc1", "fc2", "fct2"),
             "velocities":    ("boom_x_velocity", "boom_y_velocity"),
             "accelerations": ("boom_x_acceleration", "boom_y_acceleration")
             }

scalers = {}

X_features = ["torques",
              "joints",
              "velocities",
              "accelerations"]

y_features = ["ee_location",
              "joints",
              "velocities",
              "accelerations"]

X_names = list(chain(*[features[feature]
                     for feature in X_features]))

y_names = list(chain(*[features[feature]
                     for feature in y_features]))

# X_names for example could look like:
# ['fc1', 'fc2', 'fct2', 'theta1', 'theta2', 'xt2', 'boom_x_velocity', ...]


def get_feature_indices(feature_names, query_feature):
    """
    return start and end indices for feature in feature_names

    args:
        feature_names: list of features (either X_features or y_features)
        query_feature: the name of the feature whose indices you want
    returns:
        start_index
        end_index
    """

    assert query_feature in feature_names

    index = 0

    for feature in feature_names:
        if query_feature == feature:
            return index, index + len(features[feature])
        else:
            index += len(features[feature])


def normalize_data(X, y): # TODO accept prefitted scalers
    """
    normalizes inputs and outputs using MinMaxScalers

    args:
        X: input data
        y: output data
    returns:
        X: normalized input data
        y: normalized output data
    """

    scaled_X = []
    scaled_y = []

    for data, scaled_data, data_features in ((X, scaled_X, X_features),
                                             (y, scaled_y, y_features)):

        for feature in data_features:

            start_index, end_index = \
                get_feature_indices(feature_names = data_features,
                                    query_feature = feature)

            if feature in scalers: # scaler is already fitted

                scaled_data.append(
                    scalers[feature].transform(
                        data[:, start_index : end_index]))

            else: # scaler has not yet been fitted

                scalers[feature] = MinMaxScaler()

                scaled_data.append(
                    scalers[feature].fit_transform(
                        data[:, start_index : end_index]))

    X = np.concatenate(scaled_X, axis=1)
    y = np.concatenate(scaled_y, axis=1)

    X = torch.tensor(X, dtype=dtype)
    y = torch.tensor(y, dtype=dtype)

    return X, y


def load_data(path):
    """
    loads a CSV data file from path into a pandas DataFrame

    args:
        path: path to CSV file
    returns:
        data from file as a pandas DataFrame
    """

    return pd.read_csv(path)


def load_data_directory(path):
    """
    loads a directory of trajectories and separate them into testing & training

    args:
        path: Path to data directory
    returns:
        X_train: GP training inputs
        X_test: GP testing inputs
        y_train: GP training outputs
        y_test: GP testing outputs
    """

    files = os.listdir(path)

    csv_files = [file for file in files if file.endswith('.csv')]

    combined_df = []

    iteration = 0

    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        df = pd.read_csv(file_path)
        combined_df.append(df)
        iteration += 1

    combined_df = pd.concat(combined_df, ignore_index=True)

    X, y = get_xy(combined_df)

    def non_shuffling_train_test_split(X, y, test_size=0.20):
        i = int((1 - test_size) * X.shape[0]) + 1
        X_train, X_test = np.split(X, [i])
        y_train, y_test = np.split(y, [i])
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = \
        non_shuffling_train_test_split(X, y, test_size=0.20)

    X_train, y_train = normalize_data(X_train.values, y_train.values)

    X_test, y_test = normalize_data(X_test.values, y_test.values)

    return (X_train,
            X_test,
            y_train,
            y_test)


def get_xy(data):
    """
    separates a pandas DataFrame into GP inputs & outputs

    args:
        data: the DataFrame, including all values
    returns:
        X: GP inputs (joint values, torques)
        y: GP outputs (next end-effector location, next joint values)
    """

    try:

        X = data[X_names]
        y = data[y_names]

        # shift data to predict the next state

        X = X.iloc[:-1]

        y = y.shift(-1)[:-1]

        return X, y

    except Exception:
        raise AttributeError("Invalid data format")


def load_training_data(data_path, normalize=True):
    """
    loads training data from data_path

    args:
        data_path: path to trajectory file
        normalize (bool): whether data training data should be normalized
    returns:
        X: GP training inputs
        y: GP training outputs
    """

    training_data = load_data(data_path)

    X, y = get_xy(training_data)

    if normalize:

        X, y = normalize_data(X.values, y.values)

    else:

        X = torch.tensor(X.values, dtype=torch.double)
        y = torch.tensor(y.values, dtype=torch.double)

    return X, y


def load_testing_data(data_path, normalize=False):
    """
    loads testing data from data_path

    args:
        data_path: path to trajectory file
        normalize (bool): should data testing data be normalized or not
    returns:
        X: GP testing inputs
        y: GP testing outputs
    """

    testing_data = load_data(data_path)

    X, y = get_xy(testing_data)

    if normalize:
        X, y = normalize_data(X.values, y.values)

    else:
        X = torch.tensor(X.values, dtype=torch.double)
        y = torch.tensor(y.values, dtype=torch.double)

    return X, y


def plot_X_train_vs_time(X, names):
    """
    plots dataset values against time, each column in a separate subplot

    args:
        X: data to plot
        names: column names
    """

    # Create a time axis based on the number of data points
    num_data_points = X.shape[0]
    time_axis = np.arange(num_data_points)

    # Get the number of features in X
    num_features = X.shape[1]
    # Set up a single figure for all subplots
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 2 * num_features))
    fig.suptitle('Testing outputs', fontsize=24)

    # Plot each feature in X against time in subplots
    for feature_index in range(num_features):
        axs[feature_index].plot(time_axis, X[:, feature_index])
        axs[feature_index].set_xlabel('Time')
        axs[feature_index].set_ylabel(f'{names[feature_index]}')
        axs[feature_index].set_title(f'{names[feature_index]} vs. Time')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # training_path = "data/some_chill_trajectories/trajectory12_10Hz.csv"
    # data_path = "data/some_chill_trajectories/trajectory17_10Hz.csv"
    training_path = "data/two-joint_trajectories_10Hz/trajectory2.csv"
    testing_path = "data/two-joint_trajectories_10Hz/trajectory3.csv"

    data_directory = 'data/two-joint_trajectories_10Hz'

    X_train, y_train = load_training_data(training_path, True)
    X_test, y_test = load_testing_data(testing_path, True)

    X_train, X_test, y_train, y_test = load_data_directory(data_directory)
    print(X_train.shape, X_test.shape)

    plot_X_train_vs_time(X_train, X_names)
    plot_X_train_vs_time(X_test, X_names)
    plot_X_train_vs_time(y_train, y_names)
    plot_X_train_vs_time(y_test, y_names)
