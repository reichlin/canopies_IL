import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        dataset = np.load(file_path)

        ee_pos = dataset[name2arr['ee_pos']]
        obj_poses = dataset[name2arr['obj_poses']]
        J = dataset[name2arr['joints_pos']]
        J_dot = dataset[name2arr['joints_vel']]
        a = dataset[name2arr['vr_act']]

        grape_idx = np.argmin(np.sum(np.abs(ee_pos[-1] - obj_poses[0]), -1))
        g = obj_poses[:, grape_idx]

        stack_J, stack_J_dot = [], []
        for i in range(n_frames):
            idx = np.concatenate((np.array([0]*i, dtype=int), np.arange(J.shape[0]-i)))
            stack_J.append(J[idx])
            stack_J_dot.append(J_dot[idx])
        stack_J = np.concatenate(stack_J, -1)
        stack_J_dot = np.concatenate(stack_J_dot, -1)

        states.append(np.concatenate((stack_J, stack_J_dot), -1)[:-1])
        goals.append(g[:-1])
        actions.append(a[:-1])
        next_states.append(np.concatenate((stack_J, stack_J_dot), -1)[1:])

    states = torch.from_numpy(np.concatenate(states, 0)).float()
    goals = torch.from_numpy(np.concatenate(goals, 0)).float()
    actions = torch.from_numpy(np.concatenate(actions, 0)).float()
    next_states = torch.from_numpy(np.concatenate(next_states, 0)).float()

    return (states, goals, actions, next_states)



    #     # collect inputs
    #     inputs_list = []
    #     traj_len = dataset[name2arr['ee_pos']].shape[0]
    #
    #     for i in input_ids:
    #         if i == 'object_pos':
    #             ee_pos = dataset[name2arr['ee_pos']]
    #             obj_poses = dataset[name2arr['obj_poses']]
    #             grape_idx = np.argmin(np.sum(np.abs(ee_pos[-1] - obj_poses[0]), -1))
    #             target_pos = obj_poses[:, grape_idx]
    #             # obj_poses = np.array(np.split(obj_poses,obj_poses.shape[1]/3))
    #             # target_pos = get_closest_obj(obj_poses, ee_pos[-1]).flatten()
    #             input_i = target_pos
    #         else:
    #             input_arr = (dataset[name2arr[i]])
    #             input_stack = deque([input_arr[0]] * n_frames, maxlen=n_frames)
    #             input_i_stack = []
    #             for in_i in input_arr:
    #                 input_stack.append(in_i)
    #                 input_stack_i = np.array(input_stack).flatten()
    #                 input_i_stack.append(input_stack_i)
    #             input_i = np.array(input_i_stack)
    #
    #         # append
    #         inputs_list.append(input_i)
    #
    #         # collect the outputs (labels)
    #     outputs_list = []
    #     for i in output_ids:
    #         if i == 'object_pos':
    #             ee_pos = dataset[name2arr['ee_pos']]
    #             obj_poses = dataset[name2arr['obj_poses']]
    #             grape_idx = np.argmin(np.sum(np.abs(ee_pos[-1] - obj_poses[0]), -1))
    #             target_pos = obj_poses[:, grape_idx]
    #             obj_pos = get_closest_obj(dataset[name2arr['obj_poses']], ee_pos).flatten()
    #             output_i = np.tile(obj_pos, (target_trajectory_len, 1))
    #         else:
    #             output_i = dataset[name2arr[i]]
    #
    #         n = len(output_i)
    #         if i == 'vr_act':
    #             # Pad the array with zeros at the end
    #             n_pad = (n_freq - n % n_freq) % n_freq
    #             padded_array = np.pad(output_i, (0, n_pad), mode='constant')
    #             output_i = [sum(output_i[j:j + n_freq]) for j in range(0, padded_array.shape[0], n_freq)]
    #         else:
    #             output_i = [output_i[j] for j in range(0, n, n_freq)]
    #
    #         outputs_list.append(output_i)
    #
    #     # Concatenate pos, vel, and grape_pos_flat to form input
    #     inputs_list[0] = inputs_list[0][:len(inputs_list[1])]
    #     inputs = np.concatenate(inputs_list, axis=1)
    #     labels = np.concatenate(outputs_list, axis=1)
    #
    #     all_inputs.append(inputs)
    #     all_labels.append(labels)
    #
    # inputs = np.concatenate(all_inputs, axis=0)
    # labels = np.concatenate(all_labels, axis=0)
    #
    # print(f'Loaded {len(all_inputs)} trajectories (out of {len(data_files)}) for a tot of {labels.shape[0]} samples!')
    #
    # inputs_train, inputs_test, labels_train, labels_test = inputs, inputs, labels, labels
    #
    # # Convert data into PyTorch tensors
    # X_train_tensor = torch.tensor(inputs_train, dtype=torch.float32)
    # y_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
    # X_test_tensor = torch.tensor(inputs_test, dtype=torch.float32)
    # y_test_tensor = torch.tensor(labels_test, dtype=torch.float32)
    #
    # return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor



















