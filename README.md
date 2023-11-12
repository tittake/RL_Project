# Robotics Project Work course
## Reinforcement Learning task

This is an initial attempt to train a MultitaskGP model on trajectories
including all our robots' joint states.

Training can be invoked as follows:

`python train_GP_model.py trajectory1.csv trajectory2.csv`

The two required arguments are the paths to the training and testing
trajectories, respectively.

By default, the first 800 datapoints in each trajectory are read, due to memory
constraints. Modify this by changing the `MAX_DATAPOINTS` constant. Configure
the number of training iterations desired by setting `TRAINING_ITERATIONS`.

Each CSV trajectory is expected to contain the following keys:

- time
- boom_x
- boom_y
- boom_z
- boom_angle
- boom_x_velocity
- boom_y_velocity
- boom_z_velocity
- boom_x_acceleration
- boom_y_acceleration
- boom_z_acceleration
- theta1
- theta2
- fc1
- fc2
- fct2
