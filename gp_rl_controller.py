from training.GPController import GPModel
from RL.controller import RLController, PolicyNetwork

train_path = "data/some_trajectories/trajectory2_10Hz.csv"
test_path = "data/some_trajectories/trajectory1_10Hz.csv"
training_iter = 250
num_tasks = 3
ard_num_dims = 3
batch_size = 16  # Updated batch size

#Initial RL testing values
#x_boom = 2.1 y_boom = 3.5
#x_boom = 3.0 y_boom = 2.4

"""Collective controller for GP model and RL controller"""
def main():
    gpmodel = GPModel()
    gpmodel.initialize_model(train_path, test_path, num_tasks, ard_num_dims)
    gpmodel.train(training_iter)
    gpmodel.plot_training_results()
    
    

if __name__ == "__main__":
    main()