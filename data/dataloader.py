import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def normalize_data(X1, y1, mean, std, epsilon=1e-8):
    X = np.zeros(X1.shape)
    #X[:,0:1] = np.divide(X1[:, 1:] - mean, std+epsilon)
    #X[:,1:] = X1[:, 1:]

    #X = np.divide(X1 -mean, std+epsilon)
    #y = np.divide(y1, std+epsilon)
    
    scaler = StandardScaler()
    X[:,0:1] = scaler.fit_transform(X1[:, 0:1])
    X[:,1:] = X1[:, 1:]
    y = scaler.transform(y1)

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    return X, y

def calculate_mean_std(X):
    mean = np.mean(X[:, :1], axis=0)
    std = np.std(X[:, :1], axis=0)
    #mean = np.mean(X)
    #std = np.std(X)

    return mean, std

def load_data(path):
    return pd.read_csv(path)

def get_xy(data):
    try:
        #X = data[["theta1", "theta2", "xt2", "fc1", "fc2", "fct2"]]
        #y = data[["theta1", "theta2", "xt2"]]
        X = data[["theta1","fc1"]]
        y = data[["theta1"]]

        y = y - y.shift(1) 

        # discard first reading because no delta yet
        X = X.iloc[1:]
        y = y.iloc[1:]

        return X, y
    except:
        raise AttributeError("Invalid data format")

def load_training_data(train_path, normalize=True):
    train_data = load_data(train_path)
    X, y = get_xy(train_data)  
    print(X)
    print(y)
    mean_states, std_states = calculate_mean_std(X.values)
    if normalize:
        X, y = normalize_data(X.values, y.values, mean_states, std_states)
    else:
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

    return X, y 

def load_test_data(test_path, normalize=False):
    test_data = load_data(test_path)
    X, y = get_xy(test_data)

    mean_states, std_states = calculate_mean_std(X.values)
    if normalize:
        X, y = normalize_data(X.values, y.values, mean_states, std_states)
    else:
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
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
    train_path = "data/training2_short.csv"
    test_path = "data/testing2_short.csv"
    
    X_train, y_train = load_training_data(train_path, True)

    X_test, y_test = load_test_data(test_path, True)

    print(X_test)
    print(y_train)
    plot_X_train_vs_time(X_test)


    


