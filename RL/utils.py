import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import os
import random

DEFAULT_DEVICE = torch.device("cuda:0")
DEFAULT_DTYPE = torch.float32


def get_tensor(data, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)


def plot_policy(controller, model, trial=1):
    # TODO: better definitions for inputs and actions
    n_x = 10
    num_states = len(model) - 1
    inputs = np.linspace(model[0:num_states], n_x)
    actions = controller(get_tensor(inputs.reshape(-1, num_states)))

    ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Policy Plot trial {trial}")
    ax.plot(inputs, actions)
    
    
def generate_initial_values(folder_path):
    """generate initial state for RL training"""
    
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return None

    # List all files in the folder
    files = os.listdir(folder_path)
    
    csv_files = [file for file in files if file.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in the folder {folder_path}.")
        return None

    columns_to_extract = ["boom_x", "boom_y", "theta1", "theta2", "xt2"]
    extracted_values = []

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.DictReader(file)
            try:
                second_row = next(csv_reader)
                values = {column: float(second_row[column]) for column in columns_to_extract}
                extracted_values.append(values)
            except StopIteration:
                print(f"File {csv_file} is empty")

    return random.choice(extracted_values)

