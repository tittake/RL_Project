import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

scaler = StandardScaler()
scaler_y = StandardScaler()
min_max_scaler = MinMaxScaler()
min_max_y_scaler = MinMaxScaler()

X_names = ["theta1", "theta2", "xt2","fc1", "fc2", "fct2"]
y_names = ["boom_x", "boom_y", "boom_angle"]

def normalize_data(X1, y1, mean, std, test, epsilon=1e-8):
    X = np.zeros(X1.shape)
    
    #MinMax [-1,1] torques
    #MinMax for states
    if not test:
        #X[:,0:2] = min_max_scaler.fit_transform(X1[:, 0:2])
        #X[:,2:] = X1[:, 2:]
        X = min_max_scaler.fit_transform(X1)
        y = min_max_y_scaler.fit_transform(y1)
        
    else:
        #X[:,0:2] = min_max_scaler.transform(X1[:, 0:2])
        #X[:,2:] = X1[:,2:]
        X = min_max_scaler.fit_transform(X1)
        y = min_max_y_scaler.fit_transform(y1)
        #X = min_max_scaler.fit_transform(X1)
        #y = min_max_y_scaler.fit_transform(y1)

    #y = scaler.fit_transform(y1)

    X = torch.tensor(X, dtype=torch.double)
    y = torch.tensor(y, dtype=torch.double)
    return X, y

def calculate_mean_std(X):
    
    mean = np.mean(X[:, :-1], axis=0)
    std = np.std(X[:, :-1], axis=0)
    
    #mean = np.mean(X)
    #std = np.std(X)

    return mean, std

def load_data(path):
    return pd.read_csv(path)

def load_data_directory(path):
    files = os.listdir(path)

    csv_files = [file for file in files if file.endswith('.csv')]
    
    combined_df = []
    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        df = pd.read_csv(file_path)
        combined_df.append(df)

    combined_df = pd.concat(combined_df, ignore_index=True)
    
    X,y = get_xy(combined_df)

    def non_shuffling_train_test_split(X, y, test_size=0.2):
        i = int((1 - test_size) * X.shape[0]) + 1
        X_train, X_test = np.split(X, [i])
        y_train, y_test = np.split(y, [i])
        return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, y, test_size=0.2)

    mean_states_train, std_states_train = calculate_mean_std(X_train.values)
    mean_states_test, std_states_test = calculate_mean_std(X_test.values)
    
    X_train, y_train = normalize_data(X_train.values, y_train.values, mean_states_train, std_states_train, False)
    X_test, y_test = normalize_data(X_test.values, y_test.values, mean_states_test, std_states_test, True)

    
    return X_train, X_test, y_train, y_test

def get_xy(data):
    try:
        X = data[["theta1", "theta2", "xt2","fc1", "fc2", "fct2"]]
        X = data[["theta1", "theta2", "xt2"]]
        y = data[["boom_x","boom_y"]]
        
        #y = data[["x_boom", "y_boom"]]

        #y = y - y.shift(1) 
        #X = X.iloc[:-1]
        #y = y.iloc[1:]

        return X, y
    
    except Exception:
        raise AttributeError("Invalid data format")

def load_training_data(train_path, normalize=True):
    train_data = load_data(train_path)
    X, y = get_xy(train_data)  

    mean_states, std_states = calculate_mean_std(X.values)
    if normalize:
        X, y = normalize_data(X.values, y.values, mean_states, std_states, False)
    else:
        X = torch.tensor(X.values, dtype=torch.double)
        y = torch.tensor(y.values, dtype=torch.double)

    return X, y 

def load_test_data(test_path, normalize=False):
    test_data = load_data(test_path)
    X, y = get_xy(test_data)

    mean_states, std_states = calculate_mean_std(X.values)
    if normalize:
        X, y = normalize_data(X.values, y.values, mean_states, std_states, True)
    else:
        X = torch.tensor(X.values, dtype=torch.double)
        y = torch.tensor(y.values, dtype=torch.double)
    return X, y


def plot_X_train_vs_time(X, names):
    # Create a time axis based on the number of data points
    num_data_points = X.shape[0]
    time_axis = np.arange(num_data_points)

    # Get the number of features in X
    num_features = X.shape[1]
    # Set up a single figure for all subplots
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 2*num_features))
    fig.canvas.set_window_title('Scenario 4: testing outputs')

    # Plot each feature in X against time in subplots
    for feature_index in range(num_features):
        axs[feature_index].plot(time_axis, X[:, feature_index])
        axs[feature_index].set_xlabel('Time')
        axs[feature_index].set_ylabel(f'{names[feature_index]}')
        axs[feature_index].set_title(f'{names[feature_index]} vs. Time')


    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    #train_path = "data/some_chill_trajectories/trajectory12_10Hz.csv"
    #test_path = "data/some_chill_trajectories/trajectory17_10Hz.csv"
    train_path = "data/two-joint_trajectories_10Hz/trajectory2.csv"
    test_path = "data/two-joint_trajectories_10Hz/trajectory3.csv"
    
    data_directory = 'data/two-joint_trajectories_10Hz'
    X_train, y_train = load_training_data(train_path, True)
    X_test, y_test = load_test_data(test_path, True)

    X_train, X_test, y_train, y_test = load_data_directory(data_directory)
    print(X_train.shape, X_test.shape)
    
    plot_X_train_vs_time(X_train, X_names)
    plot_X_train_vs_time(X_test, X_names)
    plot_X_train_vs_time(y_train, y_names)
    plot_X_train_vs_time(y_test, y_names)


    


