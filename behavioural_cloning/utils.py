import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import yaml
from argparse import Namespace
import os

arr2name = {
    'arr_0':'joints_pos',
    'arr_1':'joints_vel',
    'arr_2':'ee_pos',
    'arr_3':'ee_or',
    'arr_4':'command_vel',
    'arr_5':'box_poses',
    'arr_6':'vr_act'
}

name2arr = {
    'joints_pos':'arr_0',
    'joints_vel':'arr_1',
    'ee_pos':'arr_2',
    'ee_or':'arr_3',
    'command_vel':'arr_4',
    'box_poses':'arr_5',
    'vr_act':'arr_6'
}

import shutil


def load_data(data_path, input_ids=['object_pos','ee_pos'], output_ids=['joints_pos'], task=None):
    # Check if the path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path '{data_path}' does not exist.")

    # Load the dataset from .npz files in the given path
    data_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

    if not data_files:
        raise FileNotFoundError("No .npz files found in the specified path.")

    # Initialize lists to store inputs and labels from all files
    all_inputs = []
    all_labels = []

    # Load data from each .npz file
    for file_name in data_files:
        
        file_path = os.path.join(data_path, file_name)
        dataset = np.load(file_path)

        #collect inputs
        inputs_list = []
        traj_len = dataset[name2arr['ee_pos']].shape[0]
        for i in input_ids:
            if i == 'object_pos':
                ee_pos = dataset[name2arr['ee_pos']]
                obj_pos = get_closest_obj(dataset[name2arr['box_poses']],ee_pos).flatten()   
                target_pos = obj_pos
                input_i = np.tile(obj_pos, (traj_len, 1))
            elif i == 'last_ee_pos':
                ee_pos = dataset[name2arr['ee_pos']]
                last_ee_pos = ee_pos[-1]
                target_pos = last_ee_pos
                input_i = np.tile(last_ee_pos, (traj_len, 1))
            else:
                input_i = (dataset[name2arr[i]])
            inputs_list.append(input_i)  

        #collect the outputs (labels)
        outputs_list = []
        for j in output_ids:
            if j == 'object_pos':
                ee_pos = dataset[name2arr['ee_pos']]
                obj_pos = get_closest_obj(dataset[name2arr['box_poses']],ee_pos).flatten()   
                outputs_list.append(np.tile(obj_pos, (traj_len, 1)))
            else:
                outputs_list.append(dataset[name2arr[j]])
        
        # Concatenate pos, vel, and grape_pos_flat to form input
        inputs = np.concatenate(inputs_list, axis=1)
        labels = np.concatenate(outputs_list, axis=1)

        all_inputs.append(inputs)
        all_labels.append(labels)
    
    inputs = np.concatenate(all_inputs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f'Loaded {len(all_inputs)} trajectories (out of {len(data_files)}) for a tot of {labels.shape[0]} samples!')
    print(f'Target posiotion is {target_pos}')
    # Split the data into training and testing sets
    #inputs_train, inputs_test, labels_train, labels_test = train_test_split(
    #    inputs, labels, test_size=0.0, random_state=42
    #)
    inputs_train, inputs_test, labels_train, labels_test =  inputs, inputs, labels, labels

    # Convert data into PyTorch tensors
    X_train_tensor = torch.tensor(inputs_train,dtype=torch.float32)
    y_train_tensor = torch.tensor(labels_train,dtype=torch.float32)
    X_test_tensor = torch.tensor(inputs_test,dtype=torch.float32)
    y_test_tensor = torch.tensor(labels_test,dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def create_data_loader(X, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_hyperparameters(config_path):
    with open(config_path, "r") as f:
        hyperparameters = yaml.safe_load(f)
    return Namespace(**hyperparameters)

def get_closest_obj(box_poses, ee_pos):
    last_ee_pos = ee_pos[-1]
    object_poses = np.array([[(g[0,0]+g[0,1])/2, (g[1,0]+g[1,1])/2, (g[2,0]+g[2,1])/2] for g in box_poses])
    closest_object_pos = object_poses[np.argmin(np.linalg.norm(object_poses - last_ee_pos, axis=1))]
    return closest_object_pos

def get_last_checkpoint(folder_path,task):
    highest_ep = float('-inf')  
    if f'model_{task}.pth' in os.listdir(folder_path):
        highest_file = f'model_final.pth'
    else:
        highest_file = None

    for filename in os.listdir(folder_path):
        if filename.startswith('model_ep_') and filename.endswith('.pth'):
            ep_number = int(filename.split('_')[2])
            if ep_number > highest_ep:
                highest_ep = ep_number
                highest_file = filename
    if highest_file:
        print(f"The file with the highest Epoch number is: {highest_file}")
        return os.path.join(folder_path,highest_file)

    else:
        print("No files found with the specified pattern.")
        return None

def get_best_model(folder_path,task):
    highest_score = float('-inf')  
    if f'model_{task}.pth' in os.listdir(folder_path):
        highest_file = f'model_final.pth'
    else:
        highest_file = None
        
    for filename in os.listdir(folder_path):
        if filename.startswith(f'{task}_model_ep_') and filename.endswith('.pth'):
            score = int(filename.split('_')[4].split('.')[0])
            if score > highest_score:
                highest_score = score
                highest_file = filename
    if highest_file:
        print(f"Loading: {highest_file}")
        return os.path.join(folder_path,highest_file)

    else:
        print("No files found with the specified pattern.")
        return None
   

def rename_traj_data(data_path,tag:str):
    files = os.listdir(data_path)
    for filename in files:
        if filename.startswith('trj_n=push_') and filename.endswith('.npz'):
            number = filename.split('_')[2].split('.')[0]
            new_filename = f'{tag}_traj_{number}.npz'
            os.rename(os.path.join(data_path, filename), os.path.join(data_path, new_filename))
            print(f"Renamed {filename} to {new_filename}")