import numpy as np
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def get_xy(data):
    try:
        #Fix
        X = data[["time","theta1", "theta2", "xt2"]] 
        y = data[["x_boom", "y_boom", "fc1", "fc2", "fct2"]]
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

    print(X_train)
    print(y_train)