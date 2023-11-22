import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
scaler_y = StandardScaler()
min_max_scaler = MinMaxScaler()
min_max_y_scaler = MinMaxScaler()

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
        #X = min_max_scaler.transform(X1)
        #y = min_max_y_scaler.transform(y1)
        X = min_max_scaler.fit_transform(X1)
        y = min_max_y_scaler.fit_transform(y1)

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

def get_xy(data):
    try:
        #X = data[["theta1", "theta2", "xt2", "fc1", "fc2", "fct2"]]
        #y = data[["theta1", "theta2", "xt2"]]
        X = data[["theta1", "theta2", "xt2", "theta1_dot", "theta2_dot", "xt2_dot"]]
        y = data[["x_boom", "y_boom"]]

        y = y - y.shift(1) 

        X = X.iloc[:-1]

        y = y.iloc[1:]

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

def plot_X_train_vs_time(X_train):
    # Create a time axis based on the number of data points
    num_data_points = X_train.shape[0]
    time_axis = np.arange(num_data_points)

    # Plot each feature in X_train against time
    for feature_index in range(X_train.shape[1]):
        plt.figure()
        plt.plot(time_axis, X_train[:, feature_index].numpy())
        plt.xlabel('Time')
        plt.ylabel(f'Feature {feature_index + 1}')
        plt.title(f'Feature {feature_index + 1} vs. Time')

        # You can save the plot as an image if needed
        # plt.savefig(f'feature_{feature_index + 1}_vs_time.png')

        plt.show()

if __name__ == "__main__":
    train_path = "data/training1_simple_10Hz.csv"
    test_path = "data/testing1_simple_10Hz.csv"
    
    X_train, y_train = load_training_data(train_path, True)

    X_test, y_test = load_test_data(test_path, True)

    plot_X_train_vs_time(X_train)
    plot_X_train_vs_time(X_test)
    plot_X_train_vs_time(y_train)
    plot_X_train_vs_time(y_test)


    


