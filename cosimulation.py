"""MATLAB/Simulink cosimulation"""

import socket
import struct
import yaml

import matlab.engine

from GP.GpModel import GpModel
from RL.Policy import RlPolicy

HOST = '127.0.0.1'
PORT = 12345


def callback(self, state: dict):
    """
    accepts a state from `RlPolicy.simulate_random_trajectory()` and sends
    its joint configuration to the MATLAB Simulink model via TCP/IP socket
    """

    joint_array = state["joints"].cpu().detach().numpy()

    joint_array = joint_array.flatten()

    joint_tensor = (self.scalers["joints"]
                    .inverse_transform(joint_array.reshape(1, -1)))

    joint_tensor = joint_tensor[0]

    joint1, joint2, joint3 = joint_tensor[:3]

    print("joints:", joint1, joint2, joint3)

    try:

        data = struct.pack('!ddd', joint1, joint2, joint3)

        matlab_socket.sendall(data)

        print("Joints sent successfully.")

    except Exception as exception:
        print(f"Error connecting to MATLAB: {exception}")


with open("configuration.yaml") as configuration_file:

    configuration = yaml.load(configuration_file,
                              Loader = yaml.SafeLoader)

matlab.engine.start_matlab()

matlab_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

matlab_socket.connect((HOST, PORT))

gp_model = GpModel(data_path        = configuration["data_path"],
                   saved_model_path = configuration["GP"]["model_path"])

rl_policy = RlPolicy(gp_model         = gp_model,
                     data_path        = configuration["data_path"],
                     saved_model_path = configuration["RL"]["model_path"])

rl_policy.simulate_random_trajectory(
    iterations      = configuration["RL"]["iterations"],
    online_learning = True,
    callback        = callback)

matlab_socket.close()
