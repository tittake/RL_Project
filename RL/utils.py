import matplotlib.pyplot as plt
import numpy as np
import torch

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
