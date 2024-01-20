import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch

min_max_joint_scaler = MinMaxScaler()
min_max_torque_scaler = MinMaxScaler()
min_max_ee_location_scaler = MinMaxScaler()

X_names = ["theta1", "theta2", "xt2", "fc1", "fc2", "fct2"]
y_names = ["boom_x", "boom_y", "theta1", "theta2", "xt2"]


def normalize_data(X, y, testing):
    '''
    Normalizes inputs and outputs with MinMaxScalers.
    Args:
        X: input data
        y: output data
        testing (bool): is X and y for training or testing
    Returns:
        X: normalized input data
        y: normalized output data
        min_max_joint_scaler: scaler for joint values
        min_max_torque_scaler: scaler for torques
        min_max_ee_location_scaler: scaler for end-effector location
    '''
    if not testing:

        joints = min_max_joint_scaler.fit_transform(X[:, 0:3])

        torques = min_max_torque_scaler.fit_transform(X[:, 3:])

        X = np.concatenate((joints, torques), axis=1)

        ee_location = min_max_ee_location_scaler.fit_transform(y[:, 0:2])

        y = np.concatenate((ee_location, joints), axis=1)

    else:

        joints = min_max_joint_scaler.transform(X[:, 0:3])

        torques = min_max_torque_scaler.transform(X[:, 3:])

        X = np.concatenate((joints, torques), axis=1)

        ee_location = min_max_ee_location_scaler.transform(y[:, 0:2])

        y = np.concatenate((ee_location, joints), axis=1)

    X = torch.tensor(X, dtype=torch.double)
    y = torch.tensor(y, dtype=torch.double)

    return (X,
            y,
            min_max_joint_scaler,
            min_max_torque_scaler,
            min_max_ee_location_scaler)


def load_data(path):
    '''
    loads datafile from path to pandas dataframe

    Args:
        path: Path to datafile
    Returns:
        Datafile in a dataframe
    '''
    return pd.read_csv(path)


def load_data_directory(path):
    '''
    loads a directory of trajectories and separate them into testing & training

    Args:
        path: Path to data directory
    Returns:
        X_train: GP training inputs
        X_test: GP testing inputs
        y_train: GP training outputs
        y_test: GP testing outputs
        joint_scaler: scaler for joint values
        torque_scaler: scaler for torques
        ee_location_scaler: scaler for end-effector locations
    '''

    files = os.listdir(path)

    csv_files = [file for file in files if file.endswith('.csv')]

    combined_df = []

    iteration = 0

    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        df = pd.read_csv(file_path)
        if iteration % 3 == 1:
            combined_df.append(df)
        iteration += 1

    combined_df = pd.concat(combined_df, ignore_index=True)

    X, y = get_xy(combined_df)

    def non_shuffling_train_test_split(X, y, test_size=0.2):
        i = int((1 - test_size) * X.shape[0]) + 1
        X_train, X_test = np.split(X, [i])
        y_train, y_test = np.split(y, [i])
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = \
        non_shuffling_train_test_split(X, y, test_size=0.2)

    (X_train,
     y_train,
     joint_scaler,
     torque_scaler,
     ee_location_scaler) = normalize_data(X_train.values,
                                          y_train.values,
                                          testing=False)

    X_test, y_test, _, _, _ = normalize_data(X_test.values,
                                             y_test.values,
                                             testing=True)

    return (X_train,
            X_test,
            y_train,
            y_test,
            joint_scaler,
            torque_scaler,
            ee_location_scaler)


def get_xy(data):
    '''
    separates a dataframe into GP inputs & outputs

    Args:
        data: Dataframe including all values
    Returns:
        X: GP inputs (joint values, torques)
        y: GP outputs (next end-effector location, next joint values)
    '''

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
    '''
    loads training data from data_path

    Args:
        data_path: path to trajectory file
        normalize (bool): should data training data be normalized or not
    Returns:
        X: GP training inputs
        y: GP training outputs
        joint_scaler: scaler for joint values
        torque_scaler: scaler for torques
        ee_location_scaler: scaler for end-effector locations
    '''

    training_data = load_data(data_path)

    X, y = get_xy(training_data)

    if normalize:

        (X,
         y,
         joint_scaler,
         torque_scaler,
         ee_location_scaler) = normalize_data(X.values,
                                              y.values,
                                              testing=False)

    else:

        X = torch.tensor(X.values, dtype=torch.double)
        y = torch.tensor(y.values, dtype=torch.double)

    return X, y, joint_scaler, torque_scaler, ee_location_scaler


def load_testing_data(data_path, normalize=False):
    '''
    loads testing data from data_path

    Args:
        data_path: path to trajectory file
        normalize (bool): should data testing data be normalized or not
    Returns:
        X: GP testing inputs
        y: GP testing outputs
        joint_scaler: scaler for joint values
        torque_scaler: scaler for torques
        ee_location_scaler: scaler for end-effector locations
    '''

    testing_data = load_data(data_path)

    X, y = get_xy(testing_data)

    if normalize:
        (X,
         y,
         joint_scaler,
         torque_scaler,
         ee_location_scaler
         ) = normalize_data(X.values, y.values, testing=True)

    else:
        X = torch.tensor(X.values, dtype=torch.double)
        y = torch.tensor(y.values, dtype=torch.double)

    return X, y, joint_scaler, torque_scaler, ee_location_scaler


def plot_X_train_vs_time(X, names):
    '''
    plots dataset values against time, each column in a separate subplot

    Args:
        X: Data to plot
        names: Column names
    '''
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
