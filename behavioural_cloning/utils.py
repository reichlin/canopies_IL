import copy
import torch
import torch.nn as nn
import argparse
import numpy as np
import yaml
from argparse import Namespace
import os
from scipy.interpolate import interp1d, splrep, splev
import matplotlib.pyplot as plt
import pickle

def get_datafiles(data_path, ends_with='.npz'):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path '{data_path}' does not exist.")
    data_files = [f for f in os.listdir(data_path) if f.endswith(ends_with)]
    if not data_files:
        raise FileNotFoundError("No .npz files found in the specified path.")
    return data_files


class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        hidden = 128
        self.f = nn.Sequential(nn.Linear(input_size, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, output_size))

    def forward(self, x, g=None):
        if g is not None:
            return self.f(torch.cat([x, g], -1))
        return self.f(x)


def load_hyperparameters(config_path):
    with open(config_path, "r") as f:
        hyperparameters = yaml.safe_load(f)
    return Namespace(**hyperparameters)


def load_data(data_path, n_frames=1):
    data_files = get_datafiles(data_path)

    print(f'Loading {len(data_files)} trajectory files.')

    # Initialize lists to store inputs and labels from all files
    states, goals, actions_bc, actions_eq, next_states, positions = [], [], [], [], [], []

    # Load data from each .npz file
    for file_name in data_files:

        #get the dataset
        file_path = os.path.join(data_path, file_name)
        dataset = np.load(file_path,allow_pickle=True)

        # process the data
        ee_pos = dataset['ee_pos']
        ee_pos_spline, _ = fit_spline(ee_pos)

        #get the action to train behavioral cloning
        ee_pos, idx = get_meaningful_pos(ee_pos)
        idx_next = np.array(idx) + 1
        dee_pos_bc = ee_pos_spline[idx_next] - ee_pos
        dee_pos_eq = ee_pos[1:] - ee_pos[:-1]

        obj_poses = dataset['obj_poses'][idx]
        J = dataset['joints_pos'][idx]
        J_dot = dataset['joints_vel'][idx]
        a = dataset['vr_act'][idx]
        grape_idx = np.argmin(np.linalg.norm(obj_poses[-1] - ee_pos[-1], axis=1))
        goal = np.tile(obj_poses[-1, grape_idx], (ee_pos.shape[0], 1))

        # stack the configurations
        stack_J, stack_J_dot = [], []
        for i in range(n_frames):
            idx = np.concatenate((np.array([0]*i, dtype=int), np.arange(J.shape[0]-i)))
            stack_J.append(J[idx])
            stack_J_dot.append(J_dot[idx])
        stack_J = np.concatenate(stack_J, -1)
        stack_J_dot = np.concatenate(stack_J_dot, -1)

        # append the data
        states.append(np.concatenate((stack_J, stack_J_dot), -1)[:-1])
        goals.append(goal[:-1])
        actions_bc.append(dee_pos_bc[:-1])
        actions_eq.append(dee_pos_eq)
        next_states.append(np.concatenate((stack_J, stack_J_dot), -1)[1:])
        positions.append(ee_pos)

    states = torch.from_numpy(np.concatenate(states, 0)).float()
    goals = torch.from_numpy(np.concatenate(goals, 0)).float()
    actions_bc = torch.from_numpy(np.concatenate(actions_bc, 0)).float()
    actions_eq = torch.from_numpy(np.concatenate(actions_eq, 0)).float()
    next_states = torch.from_numpy(np.concatenate(next_states, 0)).float()
    positions = torch.from_numpy(np.concatenate(positions, 0)).float()

    return (states, goals, actions_bc, actions_eq, next_states), positions


cmap = {0: 'k', 1: 'b', 2: 'y', 3: 'g', 4: 'r'}


def load_data_new(data_path, n_frames=1, frequency:int=50):

    original_frequency=50
    f = int(original_frequency/frequency)
    if not f%1.==0.:
        raise ValueError(f"{frequency} Hz is not a multiple of the original frequency ({original_frequency} Hz).")


    data_files = get_datafiles(data_path, '.pkl')

    print(f'Loading {len(data_files)} trajectory files.')
    #    'joint_positions', 'joint_velocities', 'cartesian_positions', 'vr_commands', 'grapes_positions'

    # Initialize lists to store inputs and labels from all files
    states, goals, actions, next_states, positions = [], [], [], [], []

    # Load data from each .npz file
    for file_name in data_files:

        #get the dataset
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)

        # process the data
        ee_pos = np.array(dataset['cartesian_positions']['value'])
        ee_time = dataset['cartesian_positions']['time']
        J = np.array(dataset['joint_positions']['value'])
        J_dot = np.array(dataset['joint_velocities']['value'])
        J_time = np.array(dataset['joint_velocities']['time'])
        obj_poses = np.array(dataset['grapes_positions'])
        acts = np.array(dataset['vr_commands']['value'])
        ee_spline = interp1d(np.array(ee_time), ee_pos, kind='cubic', fill_value="extrapolate", axis=0)
        pos = ee_spline(J_time)
        d_pos = pos[1:] - pos[:-1]

        #change the frequency and augment the dataset
        j_stack, j_next_stack, j_dot_stack, j_dot_next_stack, pos_stack, d_pos_stack = [],[],[],[],[],[]

        for i in range(f):
            J_i = J[i::f]
            J_dot_i = J_dot[i::f]
            pos_i = pos[i::f]
            d_pos_i = pos_i[1:] - pos_i[:-1]

            j_stack.append(J_i[:-1])
            j_next_stack.append(J_i[1:])
            j_dot_stack.append(J_dot_i[:-1])
            j_dot_next_stack.append(J_dot_i[1:])
            pos_stack.append(pos_i[:-1])
            d_pos_stack.append(d_pos_i)

        J = np.concatenate(j_stack, 0)
        J_next = np.concatenate(j_next_stack, 0)

        J_dot = np.concatenate(j_dot_stack, 0)
        J_dot_next = np.concatenate(j_dot_next_stack, 0)

        pos = np.concatenate(pos_stack, 0)
        d_pos = np.concatenate(d_pos_stack, 0)

        #get the grapes positions
        grape_idx = np.argmin(np.linalg.norm(obj_poses - ee_pos[-1], axis=1))
        goal = np.tile(obj_poses[grape_idx], (pos.shape[0], 1))

        '''return (torch.from_numpy(J).float(),
                torch.from_numpy(d_pos).float(),
                torch.from_numpy(goal).float(),
                torch.from_numpy(J_next).float(),
                torch.from_numpy(pos).float())'''


        state = np.concatenate((J, J_dot), -1)
        state_next = np.concatenate((J_next, J_dot_next), -1)

        # append the data
        states.append(state)
        goals.append(goal)
        actions.append(d_pos)
        next_states.append(state_next)
        positions.append(pos)

    states = torch.from_numpy(np.concatenate(states, 0)).float()
    goals = torch.from_numpy(np.concatenate(goals, 0)).float()
    actions = torch.from_numpy(np.concatenate(actions, 0)).float()
    next_states = torch.from_numpy(np.concatenate(next_states, 0)).float()
    positions = torch.from_numpy(np.concatenate(positions, 0)).float()

    return (states, goals, actions, next_states), positions





def get_config(yaml_file):
    parser = argparse.ArgumentParser(description="Behavioral cloning")
    parser.add_argument("-task", "--task", type=str,help="Task to tag to.")
    parser.add_argument("--device", type=str, default= 'cuda:0' if torch.cuda.is_available() else 'cpu',help="cuda or cpu?")
    parser.add_argument("-wb","--wb_mode", type=str, default='offline', help="\'online\' or \'offline\'?")
    parser.add_argument("--wb_group", type=str, default='Null', help="\'online\' or \'offline\'?")

    config = parser.parse_args()
    with open(yaml_file, "r") as f:
        hp_dict = yaml.safe_load(f)

    for item in hp_dict:
        if not hasattr(config, item) or (hasattr(config, item) and getattr(config, item) is None):
            setattr(config, item, hp_dict[item])

    print(f'\nSummary of the parameters\n{"-" * 20}')
    for name, value in sorted(vars(config).items()):
        print(f"{name}: {value}")
    print(f'{"-" * 20}\n')

    return config


def convolve(pos):
    N = 15
    t = pos.shape[0]
    pos_ = np.stack([np.convolve(pos[:, i], np.array([1 / N] * N), 'valid') for i in range(pos.shape[1])], axis=1)
    idx = np.arange(int(N / 2), t - int(N / 2), 1)
    return pos_, idx

def fit_spline(pos):
    t = pos.shape[0]
    delta = pos[1:] - pos[:-1]
    idx = np.unique(np.nonzero(delta)[0])
    anchor_points = pos[idx]
    all_idx = np.arange(0, t, 1)
    spline = interp1d(idx, anchor_points, kind='cubic', fill_value="extrapolate", axis=0)
    pos_ = spline(all_idx)
    return pos_, all_idx

def get_meaningful_pos(pos):
    delta = pos[1:] - pos[:-1]
    idx = np.unique(np.nonzero(delta)[0])
    pos_ = pos[idx]
    return pos_, idx














if __name__ == '__main__':

    import sys
    module_path = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning'
    sys.path.append(module_path)

    from src.utils_rl import Agent


    task='grasping'

    data, positions = load_data_new(
        data_path=f'/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data/{task}',
        n_frames=1,
        frequency=10
    )