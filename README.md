# Reinforcement Learning Task

Titta Kemppi, Väinö Pollari, Casey Jones

Robotics Project Work (KONE.533)

Tampere University

## Data

Data, consisting of randomly-generated trajectories, as well as trained models,
are stored on
[OneDrive](https://tuni-my.sharepoint.com/:f:/r/personal/casey_jones_tuni_fi/Documents/Robotics%20Project%20Work?csf=1&web=1&e=zH1AgD).

Please notify us in case of any permissions or access issues.

The basic folder hierarchy is as follows:

    ├── trained_models
    └── trajectories
        ├── 1000Hz
        │   ├── all_joints
        │   ├── joint_1_only
        │   ├── joint_2_only
        │   ├── joint_3_only
        │   ├── joints_1+2
        │   ├── joints_1+3
        │   └── joints_2+3
        ├── 100Hz
        │   ├── all_joints
        │   ├── joint_1_only
        │   ├── joint_2_only
        │   ├── joint_3_only
        │   ├── joints_1+2
        │   ├── joints_1+3
        │   └── joints_2+3
        └── 10Hz
            ├── all_joints
            ├── joint_1_only
            ├── joint_2_only
            ├── joint_3_only
            ├── joints_1+2
            ├── joints_1+3
            └── joints_2+3

We found that our GP model performed best with the 10Hz data,
so we have typically used that and recommend using it.

The remaining data is provided for reference only.

The 1000Hz data is the original output from the SimuLink model.

The 100Hz and 10Hz data were downsampled from it
using [data/downsample.py](data/downsample.py).

## Usage

### Python API

For examples of how to instantiate and invoke the GP and RL models
using Python, see [train_RL.py](train_RL.py).

### Command-Line Interface

[cli.py](cli.py) provides a command-line interface for  training and testing
the GP model, and training the RL model.

Pass either `GP` or `RL` as the first subcommand,
then `train` or `test` for the GP model
or `train` for the RL model,
followed by any arguments.

#### train a GP model:

usage:

    python cli.py GP train \
                     -d / --data_path DATA_PATH \
                     [-i / --iterations ITERATIONS] \
                     [-s / --save_model_to PATH] \
                     [-p / --plot_loss PLOT_LOSS] \
                     [-t / --test TEST]

example:

    python cli.py gp train \
                     --data_path trajectories/10Hz/all_joints \
                     --save_model_to trained_models/all_joints.pth

Optional arguments are `--iterations` 

#### evaluate a pre-trained GP model on testing data:

usage:

    python cli.py GP test \
                     -m / --model_path MODEL_PATH \
                     -d / --data_path DATA_PATH \
                     [--plot PLOT]

example:

    python cli.py gp test \
                     --data_path trajectories/10Hz/all_joints \
                     --model_path trained_models/all_joints.pth

#### train an RL model:

usage:

    cli.py RL train \
              -d / --data_path DATA_PATH \
              -g / --GP_model_path GP_MODEL_PATH \
              [-t / --trials TRIALS] \
              [-i / --iterations ITERATIONS] \
              [-l / --lr / --learning_rate LEARNING_RATE]

example:

    python cli': python cli.py rl train \
      --data_path trajectories/10Hz/all_joints \
      --gp_model_path trained_models/all_joints.pth
