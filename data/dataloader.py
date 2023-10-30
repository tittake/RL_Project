import numpy as np
import pandas as pd
import torch

def normalize_data(data):
    """"TODO: Normalize data with mean and std"""
    return

def load_data(path):
    return pd.read_csv(path)

def get_xy(data):
    try:
        
        #Time?
        X = data[["theta1", "theta2", "xt2", "fc1", "fc2", "fct2", "x_boom", "y_boom"]] #States == Joint values + Control input == Torques
        #Output => State 1 state after, add torques if need be
        y = data[["theta1", "theta2", "xt2", "x_boom", "y_boom"]]
        print(X.shape)
        print(y.shape)
        #Output is the difference of expected and current state
        y = y - y.shift(1) 
        y.iloc[0] = y.iloc[1]
        #Convert to tensors
        X = torch.tensor(X.values, dtype=torch.float)
        y = torch.t(torch.tensor(y.values, dtype=torch.float))
        return X, y
    except:
        raise AttributeError("Invalid data format")

def load_training_data(train_path, normalize=False):
    train_data = load_data(train_path)
    X, y  = get_xy(train_data)  
    
    #Add normalization

    return X,y 

def load_test_data(test_path, normalize=False):
    test_data = load_data(test_path)
    X, y = get_xy(test_data)
    return X,y

if __name__=="__main__":
    train_path = "data/training1.csv"
    test_path = "data/testing1.csv"
    
    X_train, y_train = load_training_data(train_path, True)

    X_test, y_test = load_test_data(test_path, True)

    #print(X_train)
    #print(y_train)
    print(y_train)
    print(y_test)