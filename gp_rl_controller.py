from GPController import GPModel
from RL.controller import RLController, PolicyNetwork

train_path = "data/training1_simple_100Hz.csv"
test_path = "data/testing1_simple_100Hz.csv"
training_iter = 100
num_tasks = 1
batch_size = 16  # Updated batch size

"""Collective controller for GP model and RL controller"""
def main():
    gpmodel = GPModel()
    gpmodel.initialize_model(train_path, test_path, num_tasks)
    gpmodel.train(training_iter)
    gpmodel.plot_training_results()
    

if __name__ == "__main__":
    main()