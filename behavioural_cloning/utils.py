import torch
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

def load_data(data_path, input_ids=['object_pos','ee_pos'], output_ids=['joints_pos'], n_frames=1, n_freq=1,  task=None):
   
    data_files = get_datafiles(data_path)

    # Initialize lists to store inputs and labels from all files
    all_inputs = []
    all_labels = []

    # Load data from each .npz file
    for file_name in data_files:
        if file_name=='traj_imitation.npz':
            continue
        file_path = os.path.join(data_path, file_name)
        dataset = np.load(file_path)
        #collect inputs
        inputs_list = []
        traj_len = dataset[name2arr['ee_pos']].shape[0]
        target_trajectory_len = int(traj_len/n_freq + 1)

        for i in input_ids:
            if i == 'object_pos':
                ee_pos = dataset[name2arr['ee_pos']]
                obj_poses = dataset[name2arr['obj_poses']]
                grape_idx = np.argmin(np.sum(np.abs(ee_pos[-1] - obj_poses[0]), -1))
                target_pos = obj_poses[:,grape_idx]
                # obj_poses = np.array(np.split(obj_poses,obj_poses.shape[1]/3))
                # target_pos = get_closest_obj(obj_poses, ee_pos[-1]).flatten()
                input_i = np.tile(target_pos, (target_trajectory_len, 1))
            elif i == 'last_ee_pos':
                ee_pos = dataset[name2arr['ee_pos']]
                last_ee_pos = ee_pos[-1]
                target_pos = last_ee_pos
                input_i = np.tile(last_ee_pos, (target_trajectory_len, 1))
            else:
                input_arr = (dataset[name2arr[i]])
                input_stack = deque([input_arr[0]]*n_frames, maxlen = n_frames)
                input_i_stack = []
                for in_i in input_arr:
                    input_stack.append(in_i)
                    input_stack_i = np.array(input_stack).flatten()
                    input_i_stack.append(input_stack_i)
                input_i = np.array(input_i_stack)

                #bring it from 50 Hz to 1 Hz
                input_i = [input_i[j] for j in range(0,len(input_i),n_freq)]
                        
            #append
            inputs_list.append(input_i) 

        #collect the outputs (labels)
        outputs_list = []
        for i in output_ids:
            if i == 'object_pos':
                ee_pos = dataset[name2arr['ee_pos']]
                obj_pos = get_closest_obj(dataset[name2arr['obj_poses']],ee_pos).flatten()   
                output_i = np.tile(obj_pos, (target_trajectory_len, 1))
            else:
                output_i = dataset[name2arr[i]]
            
            n = len(output_i)
            if i == 'vr_act':
                # Pad the array with zeros at the end
                n_pad = (n_freq - n % n_freq) % n_freq
                padded_array = np.pad(output_i, (0, n_pad), mode='constant')
                output_i = [sum(output_i[j:j+n_freq]) for j in range(0,padded_array.shape[0],n_freq)]
            else: 
                output_i = [output_i[j] for j in range(0,n,n_freq)]
            
            outputs_list.append(output_i)


        # Concatenate pos, vel, and grape_pos_flat to form input
        inputs_list[0] = inputs_list[0][:len(inputs_list[1])]
        inputs = np.concatenate(inputs_list, axis=1)
        labels = np.concatenate(outputs_list, axis=1)

        all_inputs.append(inputs)
        all_labels.append(labels)

    inputs = np.concatenate(all_inputs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f'Loaded {len(all_inputs)} trajectories (out of {len(data_files)}) for a tot of {labels.shape[0]} samples!')

    inputs_train, inputs_test, labels_train, labels_test = inputs, inputs, labels, labels

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

def get_closest_obj(obj_poses, ee_pos):
    closest_object_pos = obj_poses[np.argmin(np.linalg.norm(obj_poses - ee_pos, axis=1))]
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


def load_trajectories(data_path, n_frames=1, n_freq=1, max_number=100,  task=None):
    data_files = get_datafiles(data_path)
    trajectories = []
    states, acts, ee_pos, dpos, next_states = [],[],[],[],[]
    cnt = 0 
    # Load data from each .npz file
    for file_name in data_files:
        if file_name=='traj_imitation.npz':
            continue
        if cnt==max_number:
            break
        #print(file_name)

        file_path = os.path.join(data_path, file_name)
        dataset = np.load(file_path)

        # get the observations(goal, joint_pos, joint_vel) and actions from the dataset
        ee_positions = dataset[name2arr['ee_pos']]
        obj_poses = dataset[name2arr['obj_poses']]
        grape_idx = np.argmin(np.sum(np.abs(ee_positions[-1] - obj_poses[0]), -1))
        target_pos = obj_poses[:, grape_idx]
        # obj_poses = np.array(np.split(obj_poses,obj_poses.shape[1]/3))
        # target_pos = get_closest_obj(obj_poses, ee_pos[-1]).flatten()
        # input_i = np.tile(target_pos, (target_trajectory_len, 1))
        # obj_poses = np.array(np.split(obj_poses,obj_poses.shape[0]/3))
        # target_pos = get_closest_obj(obj_poses, ee_positions[-1]).flatten()
        joint_poses = list(dataset[name2arr['joints_pos']])
        joint_vels = list(dataset[name2arr['joints_vel']])
        n = len(joint_poses)
        actions = list(dataset[name2arr['vr_act']])[:-1]
        ee_positions = list(ee_positions)

        #bring everything from 50Hz to 1Hz (padding if necessary)
        n_pad = (n_freq - n % n_freq) % n_freq +1
        joint_poses.extend([joint_poses[-1]]*n_pad)
        joint_vels.extend([joint_vels[-1]]*n_pad)
        ee_positions.extend([ee_positions[-1]]*n_pad)

        indices = [i for i in range(0,n+n_pad,n_freq)]
        joint_poses = [joint_poses[j] for j in indices]
        joint_vels = [joint_vels[j] for j in indices]
        ee_positions = [ee_positions[j] for j in indices]

        indices = [i for i in range(0,n,n_freq)]
        actions = [sum(actions[j:j+n_freq]) for j in indices]
        delta_pos = [ee_positions[i+1]-ee_positions[i] for i in range(len(ee_positions)-1)]

        #create the stacks of positions and observations
        pos_stack = deque([joint_poses[0]]*n_frames, maxlen = n_frames)
        vel_stack = deque([joint_vels[0]]*n_frames, maxlen = n_frames)
        pos_0 = np.array(pos_stack).flatten().squeeze()
        vel_0 = np.array(vel_stack).flatten().squeeze()
        state_0 = np.concatenate([pos_0, vel_0])

        for i in range(1,len(joint_poses)):
            #update the stacks and get the new observations and action
            pos_stack.append(joint_poses[i])
            vel_stack.append(joint_vels[i])
            pos_1 = np.array(pos_stack).flatten()
            vel_1 = np.array(vel_stack).flatten()
            state_1 = np.concatenate([pos_1, vel_1])
            
            states.append(torch.from_numpy(state_0))        #t=i-1
            ee_pos.append(torch.from_numpy(ee_positions[i-1]))
            acts.append(torch.tensor(actions[i-1]))         #t=i-1
            dpos.append(torch.tensor(delta_pos[i-1]))       #t=i-1
            next_states.append(torch.from_numpy(state_1))   #t=i

            state_0 = state_1
        cnt+=1
    return Data(states, ee_pos, acts, dpos, next_states)



def load_positions(data_path, max_number=1,  task=None):
    data_files = get_datafiles(data_path)

    trajectories = []

    cnt = 0 
    # Load data from each .npz file
    for file_name in data_files:
        if file_name=='traj_imitation.npz':
            continue
        if cnt>max_number:
            break
        print(file_name)
        file_path = os.path.join(data_path, file_name)
        dataset = np.load(file_path)

        ee_pos = list(dataset[name2arr['ee_pos']])
        n = len(ee_pos)

        #bring everything from 50Hz to 1Hz (padding if necessary)
        n_pad = (50 - n % 50) % 50 +1
        ee_pos.extend([ee_pos[-1]]*n_pad)

        indices = [i for i in range(0,n+n_pad,50)]
        ee_pos = [ee_pos[j] for j in indices]

        trajectories.extend(list(ee_pos))
        cnt+=1
       
        
    return ee_pos