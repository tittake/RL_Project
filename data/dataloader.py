import numpy as np
import pandas as pd
import torch

def normalize_data(X1, y1, mean, std, epsilon=1e-8):
    X = np.zeros(X1.shape)
    X[:,0:5] = np.divide(X1[:, :-3] - mean, std+epsilon)
    X[:,5:] = X1[:, 5:]
    y = np.divide(y1, std+epsilon)
    
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    return X, y

def calculate_mean_std(X):
    mean = np.mean(X[:, :5], axis=0)
    std = np.std(X[:, :5], axis=0)
    return mean, std

def load_data(path):
    return pd.read_csv(path)

def get_xy(data):
    try:
        X = data[["theta1", "theta2", "xt2", "x_boom", "y_boom", "fc1", "fc2", "fct2"]]
        y = data[["theta1", "theta2", "xt2", "x_boom", "y_boom"]]

        y = y - y.shift(1) 
        y.iloc[0] = y.iloc[1]
        return X, y
    except:
        raise AttributeError("Invalid data format")

def load_training_data(train_path, normalize=True):
    train_data = load_data(train_path)
    X, y = get_xy(train_data)  
    
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

if __name__ == "__main__":
    train_path = "data/training1.csv"
    test_path = "data/testing1.csv"
    
    X_train, y_train = load_training_data(train_path, True)

    X_test, y_test = load_test_data(test_path, True)

    print(X_train)
    print(y_train)
