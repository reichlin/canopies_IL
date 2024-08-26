import copy
import torch
import torch.nn as nn
import argparse
import numpy as np
import yaml
from argparse import Namespace
import os
import torch.nn.functional as F
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import normalize

def get_datafiles(data_path, ends_with='.npz'):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path '{data_path}' does not exist.")
    data_files = [f for f in os.listdir(data_path) if f.endswith(ends_with)]
    if not data_files:
        raise FileNotFoundError("No .npz files found in the specified path.")
    return data_files


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden:int=32, big=False):
        super(MLP, self).__init__()
        if big:
            self.f = nn.Sequential(nn.Linear(input_size, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output_size))
        else:
            self.f = nn.Sequential(nn.Linear(input_size, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, output_size))

    def forward(self, x, g=None):
        if g is not None:
            return self.f(torch.cat([x, g], -1))
        return self.f(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class MLPDual(nn.Module):

    def __init__(self, input_size, output_size:tuple, hidden:int = 64):
        super(MLPDual, self).__init__()
        self.f = nn.Sequential(nn.Linear(input_size, hidden),
                               nn.BatchNorm1d(hidden),
                               nn.ReLU(),
                               #nn.Dropout(0.2),
                               nn.Linear(hidden, hidden),
                               nn.BatchNorm1d(hidden),
                               nn.ReLU(),
                               #nn.Dropout(0.2),
                               nn.Linear(hidden, hidden),
                               nn.BatchNorm1d(hidden),
                               nn.ReLU(),
                               #nn.Dropout(0.2)
                               )
        self.head1 = nn.Sequential(nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output_size[0])
                                   )

        self.head2 = nn.Sequential(nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output_size[1])
                                   )

    def forward(self, x, g):
        h = self.f(torch.cat([x, g], -1))
        return torch.cat([self.head1(h), F.normalize(self.head2(h), dim=-1)], dim=-1)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))



def load_hyperparameters(config_path):
    with open(config_path, "r") as f:
        hyperparameters = yaml.safe_load(f)
    return Namespace(**hyperparameters)


def load_data_representation(data_path, frequency=10, j_only=False, single_traj=False):

    if not 50/frequency%1==0:
        raise ValueError(f"Frequency must be a dividend of 50, not {frequency}.")
    f = int(50/frequency)

    data_files = get_datafiles(data_path, '.pkl')

    print(f'Loading {len(data_files)} trajectory files for equivariance representations.')

    # Initialize lists to store inputs and labels from all files
    states, goals, actions, next_states, positions = [], [], [], [], []

    # Load data from each .npz file
    for file_name in data_files:

        #get the dataset
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)

        # get the data while
        ee_pos = np.array(dataset['cartesian_positions']['value'])
        ee_time = dataset['cartesian_positions']['time']
        J_array = np.array(dataset['joint_positions']['value'])
        J_dot_array = np.array(dataset['joint_velocities']['value'])
        J_time = np.array(dataset['joint_velocities']['time'])
        obj_poses = np.array(dataset['grapes_positions'])


        ee_pos[:5] = ee_pos[5]
        ee_spline = interp1d(np.array(ee_time), ee_pos, kind='cubic', fill_value="extrapolate", axis=0)


        all_indices = [i for i in range(J_time.shape[0])]

        for i in range(f):
            # downsample with the frequency
            indices_i = all_indices[i::f]
            J = J_array[indices_i][:-1]
            J_next = J_array[indices_i][1:]
            J_dot = J_dot_array[indices_i][:-1]
            J_dot_next = J_dot_array[indices_i][1:]
            timesteps = J_time[indices_i][:-1]
            timesteps_next = J_time[indices_i][1:]
            pos = ee_spline(timesteps)


            #compose the states (with joint vel and configurations)
            state = J if j_only else np.concatenate((J, J_dot), -1)
            state_next = J_next if j_only else np.concatenate((J_next, J_dot_next), -1)

            # get the grapes positions
            grape_idx = np.argmin(np.linalg.norm(obj_poses - ee_pos[-1], axis=1))
            goal = np.tile(obj_poses[grape_idx], (state.shape[0], 1))

            #SPLINING AND SAMPLING
            act = ee_spline(timesteps_next) - ee_spline(timesteps)

            # append the data
            states.append(state)
            goals.append(goal)
            actions.append(act)
            next_states.append(state_next)
            positions.append(pos)

    if single_traj:
        return states, goals, actions, next_states, positions
    else:
        states = torch.from_numpy(np.concatenate(states, 0)).float()
        goals = torch.from_numpy(np.concatenate(goals, 0)).float()
        actions = torch.from_numpy(np.concatenate(actions, 0)).float()
        next_states = torch.from_numpy(np.concatenate(next_states, 0)).float()
        positions = torch.from_numpy(np.concatenate(positions, 0)).float()
        return states, goals, actions, next_states, positions





def load_data(data_path, j_only=True, convolving=True, single_traj=False, frequency=50):

    if not 50 / frequency % 1 == 0:
        raise ValueError(f"Frequency must be a dividend of 50, not {frequency}.")
    f = int(50 / frequency)

    data_files = get_datafiles(data_path, '.pkl')
    print(f'Loading {len(data_files)} trajectory files.')

    states, goals, actions, next_states, positions = [], [], [], [], []
    cnt=0

    for file_name in data_files:
        cnt+=1
        #get the dataset
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)

        # get the data
        ee_pos = np.array(dataset['cartesian_positions']['value'])
        J_array = np.array(dataset['joint_positions']['value'])
        J_dot_array = np.array(dataset['joint_velocities']['value'])
        J_time = np.array(dataset['joint_velocities']['time'])
        obj_poses = np.array(dataset['grapes_positions'])
        target_rot_array = np.array(dataset['target_orientations']['value'])
        dpos_array = np.array(dataset['command_positions']['value'])

        # SPLINING AND SAMPLING
        dpos_array[:10] = dpos_array[10]

        dpos_spline = interp1d(dataset['command_positions']['time'], dpos_array, kind='cubic', fill_value="extrapolate",
                               axis=0)
        target_rot_spline = interp1d(dataset['target_orientations']['time'], target_rot_array, kind='cubic',
                                     fill_value="extrapolate", axis=0)
        ee_spline = interp1d(dataset['cartesian_positions']['time'], ee_pos, kind='cubic', fill_value="extrapolate",
                             axis=0)
        d_act_pos = dpos_spline(J_time)

        act_rot = normalize(target_rot_spline(J_time), axis=1)
        ee_pos = ee_spline(J_time)

        all_indices = [i for i in range(J_time.shape[0])]

        for i in range(f):

            # downsample with the frequency
            indices_i = all_indices[i::f]
            J = J_array[indices_i][:-1]
            J_next = J_array[indices_i][1:]
            J_dot = J_dot_array[indices_i][:-1]
            J_dot_next = J_dot_array[indices_i][1:]
            act_rot_i = act_rot[indices_i][:-1]
            pos = ee_pos[indices_i][:-1]
            timesteps = J_time[indices_i][:-1]

            # get the states
            state = J if j_only else np.concatenate((J, J_dot), -1)
            state_next = J_next if j_only else np.concatenate((J_next, J_dot_next), -1)

            # get the grapes positions
            grape_idx = np.argmin(np.linalg.norm(obj_poses - ee_pos[-1], axis=1))
            goal = np.tile(obj_poses[grape_idx], (state.shape[0], 1))

            #integrate the actions for downsampling
            d_act_pos_i = np.array([np.sum(d_act_pos[idx:idx+f], axis=0) for idx in indices_i])[:-1]

            #CONVOLVING
            if convolving:
                d_act_pos_i, idx = convolve(d_act_pos_i)
                state = state[idx]
                state_next = state_next[idx]
                goal = goal[idx]
                pos = pos[idx]
                act_rot_i = act_rot_i[idx]
                timesteps = timesteps[idx]

            act = np.concatenate((d_act_pos_i, act_rot_i), -1)

            # append the data
            states.append(state)
            goals.append(goal)
            actions.append(act)
            next_states.append(state_next)
            positions.append(pos)
    if single_traj:
        return states, goals, actions, next_states, positions
    else:
        states = torch.from_numpy(np.concatenate(states, 0)).float()
        goals = torch.from_numpy(np.concatenate(goals, 0)).float()
        actions = torch.from_numpy(np.concatenate(actions, 0)).float()
        next_states = torch.from_numpy(np.concatenate(next_states, 0)).float()
        positions = torch.from_numpy(np.concatenate(positions, 0)).float()
        return states, goals, actions, next_states, positions

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


def convolve(x):
    N = 15 #15
    t = x.shape[0]
    x_ = np.stack([np.convolve(x[:, i], np.array([1 / N] * N), 'valid') for i in range(x.shape[1])], axis=1)
    idx = np.arange(int(N / 2), t - int(N / 2), 1)
    return x_, idx

def fit_spline(pos):
    t = pos.shape[0]
    delta = pos[1:] - pos[:-1]
    idx = np.unique(np.nonzero(delta)[0])
    anchor_points = pos[idx]
    all_idx = np.arange(0, t, 1)
    spline = interp1d(idx, anchor_points, kind='cubic', fill_value="extrapolate", axis=0)
    pos_ = spline(all_idx)
    return pos_, all_idx

def filter(x, th=0.01):
    pd = np.mean(abs(x), axis=1)
    x[pd < th] = 0.
    idx = np.unique(np.nonzero(x)[0])
    return x[idx], idx

def integrate(X, T):
    x_ = np.zeros(X.shape[-1])
    t_ = T[0]
    y = [np.zeros(X.shape[-1])]
    for x, t in zip(X[:-1],T[1:]):
        dt = t - t_
        x_ += x*dt
        y.append(copy.deepcopy(x_))
        t_ = copy.deepcopy(t)
    return np.stack(y)




if __name__ == '__main__':

    import sys
    module_path = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning'
    sys.path.append(module_path)
    task='grasp_last'

    states, goals, actions, next_states, positions = load_data_representation(
        data_path=f'/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data/{task}',
        j_only=True,
        single_traj=True,
        frequency=10
    )
    plt.scatter(np.concatenate(positions[:5])[:, 1], np.concatenate(positions[:5])[:, 2])
    plt.show()




