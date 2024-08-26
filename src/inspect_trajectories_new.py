import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from utils_rl import Agent
import torch
import pickle


def plot_trajectories():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(training_points[:, 0], training_points[:, 1], training_points[:, 2], color='red', label='training traj')
    ax.scatter(stable_agent_points[:, 0], stable_agent_points[:, 1], stable_agent_points[:, 2], color='blue', label='resulting stable trajectory')
    ax.scatter(normal_agent_points[:, 0], normal_agent_points[:, 1], normal_agent_points[:, 2], color='purple', label='resulting BC trajectory')

    plt.show()



if __name__ == "__main__":
    task = 'grasping'

    data_dir = f'/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data/{task}'

    #load training traj. positions
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    positions = []
    for file_name in data_files:
        # get the dataset
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)

        pos = dataset['cartesian_positions']['value']
        positions.append(np.array(pos))

    training_points = np.concatenate(positions, axis=0)


    results_dir = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/results'

    #load recorded traj. positions
    data_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl') and f.startswith(task)]
    stable_agent_positions = []
    normal_agent_positions = []
    for file_name in data_files:
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, 'rb') as file:
            rec_traj_dict = pickle.load(file)
        pos = rec_traj_dict['cartesian_positions']['value']
        if 'stable' in file_name:
            stable_agent_positions.append(np.array(pos))
        else:
            normal_agent_positions.append(np.array(pos))
    stable_agent_points = np.concatenate(stable_agent_positions, axis=0)
    normal_agent_points = np.concatenate(normal_agent_positions, axis=0)


    plot_trajectories()
