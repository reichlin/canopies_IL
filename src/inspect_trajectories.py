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
    ax.scatter(training_points[:, 0], training_points[:, 1], training_points[:, 2], color='purple', alpha=0.01, s=10,label='training traj')
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], color='red', label='resulting stable trajectory')
    #ax.scatter(training_goals[:, 0], training_goals[:, 1], training_goals[:, 2], color='green')
    ax.scatter(agent_points[:, 0], agent_points[:, 1], agent_points[:, 2],
               s=1,
               label='resulting stable trajectory')

    plt.show()

def plot_actions():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(training_actions[:, 0], training_actions[:, 1], training_actions[:, 2],
               color='purple',
               alpha=0.1, s=10,
               label='training acts')
    ax.scatter(agent_actions[:, 0], agent_actions[:, 1], agent_actions[:, 2],
               s=1,
               label='agent acts')

    plt.show()

import copy

if __name__ == "__main__":
    task = 'grasping'

    data_dir = f'/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data/grasp_best'

    #load training traj. positions
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and f.startswith('1723127422')]
    positions, goals, actions = [],[],[]
    for file_name in data_files:
        # get the dataset
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
        obj_poses = dataset['grapes_positions']

        pos = dataset['cartesian_positions']['value']
        act = dataset['target_positions']['value']
        grape_idx = np.argmin(np.linalg.norm(np.array(obj_poses) - np.array(pos[-1]), axis=1))
        goals.append(obj_poses[grape_idx])
        actions.append(act)
        positions.append(np.array(pos))
        print(dataset['target_positions']['value'][4])


    training_points = np.concatenate(positions, axis=0)
    training_actions = np.concatenate(actions, axis=0)
    training_goals = np.stack(goals, axis=0)

    print('---')
    results_dir = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/results'

    #load recorded traj. positions
    data_files = [f for f in os.listdir(results_dir)
                  if f.endswith('.pkl') and f.startswith('prove_new') ]
    positions, goals, actions = [],[],[]


    car = 'blue'
    siium = 10
    for file_name in data_files:
        file_path = os.path.join(results_dir, file_name)
        print(file_path)

        with open(file_path, 'rb') as file:
            rec_traj_dict = pickle.load(file)
        act_ = copy.deepcopy(act)
        pos = np.array(rec_traj_dict['cartesian_positions']['value'])
        goal = rec_traj_dict['grapes_positions']
        act = np.array(rec_traj_dict['target_positions']['value'])# + np.array([0.,0.,np.random.uniform(-0.1,0.1)])

        goals.append(goal)
        actions.append(act)
        positions.append(pos)
        print(goal)
        #for i in range(3):
        #    plt.scatter(timesteps, pos[:,i], s=siium, color=car)

    goals = np.stack(goals).squeeze()
    agent_points = np.concatenate(positions, axis=0)
    agent_actions = np.concatenate(actions, axis=0)

    plot_actions()
    plot_trajectories()
    dio = 0

