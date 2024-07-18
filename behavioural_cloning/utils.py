import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import numpy as np
import yaml
from argparse import Namespace
import os
from collections import deque, namedtuple

arr2name = {
    'arr_0':'joints_pos',
    'arr_1':'joints_vel',
    'arr_2':'ee_pos',
    'arr_3':'ee_or',
    'arr_4':'command_vel',
    'arr_5':'obj_poses',
    'arr_6':'vr_act'
}

name2arr = {
    'joints_pos':'arr_0',
    'joints_vel':'arr_1',
    'ee_pos':'arr_2',
    'ee_or':'arr_3',
    'command_vel':'arr_4',
    'obj_poses':'arr_5',
    'vr_act':'arr_6'
}


Data = namedtuple('Dataset', ['states', 'ee_pos', 'actions', 'dpos', 'next_states'])


def get_datafiles(data_path):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path '{data_path}' does not exist.")
    data_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
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
                               nn.Linear(hidden, output_size))

    def forward(self, x, g=None):
        if g is not None:
            return self.f(torch.cat([x, g], -1))
        return self.f(x)


def load_hyperparameters(config_path):
    with open(config_path, "r") as f:
        hyperparameters = yaml.safe_load(f)
    return Namespace(**hyperparameters)


def load_data(data_path, n_frames=1, task=None):
    data_files = get_datafiles(data_path)

    # Initialize lists to store inputs and labels from all files
    states, goals, actions, next_states = [], [], [], []

    # Load data from each .npz file
    for file_name in data_files:

        file_path = os.path.join(data_path, file_name)
        dataset = np.load(file_path,allow_pickle=True)

        ee_pos = dataset['ee_pos']
        obj_poses = dataset['obj_poses']
        J = dataset['joints_pos']
        J_dot = dataset['joints_vel']
        a = dataset['vr_act']

        grape_idx = np.argmin(np.sum(np.abs(ee_pos[-1] - obj_poses[0]), 0))
        g = obj_poses[:, grape_idx]

        #grape_idx = np.argmin(np.linalg.norm(obj_poses - ee_pos[-1], axis=1))
        #g = obj_poses[grape_idx]
        #goal = np.tile(g, (a.shape[0], 1))

        stack_J, stack_J_dot = [], []
        for i in range(n_frames):
            idx = np.concatenate((np.array([0]*i, dtype=int), np.arange(J.shape[0]-i)))
            stack_J.append(J[idx])
            stack_J_dot.append(J_dot[idx])
        stack_J = np.concatenate(stack_J, -1)
        stack_J_dot = np.concatenate(stack_J_dot, -1)
        dee_pos = ee_pos[1:] - ee_pos[:-1]

        states.append(np.concatenate((stack_J, stack_J_dot), -1)[:-1])
        goals.append(g[:-1])
        #actions.append(a[:-1])
        actions.append(dee_pos)
        next_states.append(np.concatenate((stack_J, stack_J_dot), -1)[1:])

    states = torch.from_numpy(np.concatenate(states, 0)).float()
    goals = torch.from_numpy(np.concatenate(goals, 0)).float()
    actions = torch.from_numpy(np.concatenate(actions, 0)).float()
    next_states = torch.from_numpy(np.concatenate(next_states, 0)).float()

    return (states, goals, actions, next_states)


def get_config(yaml_file):
    parser = argparse.ArgumentParser(description="Behavioral cloning")
    parser.add_argument("-task", "--task", type=str, default="grasp_new",help="Task to tag to.")
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
