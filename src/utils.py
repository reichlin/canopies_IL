import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from collections import deque, namedtuple
import rospy

name2arr = {
    'joints_pos': 'arr_0',
    'joints_vel': 'arr_1',
    'ee_pos': 'arr_2',
    'ee_or': 'arr_3',
    'command_vel': 'arr_4',
    'obj_poses': 'arr_5',
    'vr_act': 'arr_6'
}
Data = namedtuple('Dataset', ['states', 'actions', 'next_states'])


class TrajectoryHandler:
    def __init__(self, save_dir):
        self.reset()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def reset(self):
        self.data_joints_pos = []
        self.data_joints_vel = []
        self.data_pos = []
        self.data_or = []
        self.data_joint_act = []
        self.data_vr_act = []
        self.data_external_objs = []

    def size(self):
        return len(self.data_joints_pos)

    def store_data(self, data):
        self.data_joints_pos.append(data[0])
        self.data_joints_vel.append(data[1])
        self.data_pos.append(data[2])
        self.data_or.append(data[3])
        self.data_joint_act.append(data[4])
        if data[5] is not None:
            self.data_vr_act.append(data[5])
        self.data_external_objs.append(data[6])

    def show_current_status(self):
        rospy.loginfo(f'------------- SUMMARY -------------')
        rospy.loginfo(f'joint positions: {len(self.data_joints_pos)}')
        rospy.loginfo(f'joint velocities: {len(self.data_joints_vel)}')
        rospy.loginfo(f'ee positions: {len(self.data_pos)}')
        rospy.loginfo(f'ee orientations: {len(self.data_or)}')
        rospy.loginfo(f'joint vel actions: {len(self.data_joint_act)}')
        rospy.loginfo(f'vr commands: {len(self.data_vr_act)}')
        rospy.loginfo(f'------------------------------------')

    def save_trajectory(self, name):
        if self.size() > 10:
            name = f"traj_{name}.npz"
            file = os.path.join(self.save_dir, name)
            np.savez(file,
                     joints_pos=np.concatenate(self.data_joints_pos, 0),
                     joints_vel=np.concatenate(self.data_joints_vel, 0),
                     ee_pos=np.concatenate(self.data_pos, 0),
                     ee_or=np.concatenate(self.data_or, 0),
                     command_vel=np.concatenate(self.data_joint_act, 0),
                     obj_poses=self.data_external_objs,
                     vr_act=np.concatenate(self.data_vr_act, 0),
                     )
            self.show_current_status()
            return file
        else:
            return 'NEVER'

def process_obs(obs_dict):
    joint_pos = torch.from_numpy(np.expand_dims(np.stack(obs_dict['joint_pos'], axis=0).flatten(), 0)).float()
    joint_vel = torch.from_numpy(np.expand_dims(np.stack(obs_dict['joint_vel'], axis=0).flatten(), 0)).float()
    obs_tsr = torch.cat([joint_pos, joint_vel], dim=1)
    return obs_tsr

#
# def get_closest_obj(obj_poses, ee_pos):
#     closest_object_pos = obj_poses[np.argmin(np.linalg.norm(obj_poses - ee_pos, axis=1))]
#     return closest_object_pos
#
#
# def load_trajectories(data_path, n_frames=1, n_freq=1):
#     # Check if the path exists
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Path '{data_path}' does not exist.")
#
#     data_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
#
#     if not data_files:
#         raise FileNotFoundError("No .npz files found in the specified path.")
#
#     # Initialize lists to store the trajectories in the form: (s,a,s')
#     trajectories = []
#     states, acts, next_states = [], [], []
#
#     # Load data from each .npz file
#     for file_name in data_files:
#         if file_name == 'traj_imitation.npz':
#             continue
#         file_path = os.path.join(data_path, file_name)
#         dataset = np.load(file_path)
#
#         # get the observations(goal, joint_pos, joint_vel) and actions from the dataset
#         ee_pos = dataset[name2arr['ee_pos']]
#         obj_poses = dataset[name2arr['obj_poses']]
#         grape_idx = np.argmin(np.sum(np.abs(ee_pos[-1] - obj_poses[0]), -1))
#         target_pos = obj_poses[:, grape_idx]
#         # obj_poses = np.array(np.split(obj_poses, obj_poses.shape[0] / 3))
#         # target_pos = get_closest_obj(obj_poses, ee_pos[-1]).flatten()
#         joint_poses = list(dataset[name2arr['joints_pos']])
#         joint_vels = list(dataset[name2arr['joints_vel']])
#         actions = list(dataset[name2arr['ee_pos']])
#         # bring everything from 50Hz to 1Hz (padding if necessary)
#         n = len(joint_poses)
#         n_pad = int((n_freq - n % n_freq) % n_freq)
#         joint_poses.extend([joint_poses[-1]] * n_pad)
#         joint_vels.extend([joint_vels[-1]] * n_pad)
#         actions.extend([np.zeros(3)] * n_pad)
#
#         # np.pad(actions, (0, n_pad), mode='constant')
#
#         joint_poses = [joint_poses[j] for j in range(0, n, n_freq)]
#         joint_vels = [joint_vels[j] for j in range(0, n, n_freq)]
#         actions = [sum(actions[j:j + n_freq]) for j in range(0, n + n_pad, n_freq)]
#
#         # create the stacks of positions and observations
#         pos_stack = deque([joint_poses[0]] * n_frames, maxlen=n_frames)
#         vel_stack = deque([joint_vels[0]] * n_frames, maxlen=n_frames)
#         pos_0 = np.array(pos_stack).flatten().squeeze()
#         vel_0 = np.array(vel_stack).flatten().squeeze()
#         state_0 = np.concatenate([target_pos[0], pos_0, vel_0])
#
#         for pos_i, vel_i, act_i in zip(joint_poses, joint_vels, actions):
#             # update the stacks and get the new observations and action
#             pos_stack.append(pos_i)
#             vel_stack.append(vel_i)
#             pos_1 = np.array(pos_stack).flatten()
#             vel_1 = np.array(vel_stack).flatten()
#             state_1 = np.concatenate([target_pos[0], pos_1, vel_1])
#
#             states.append(torch.from_numpy(state_0))
#             acts.append(torch.tensor(act_i))
#             next_states.append(torch.from_numpy(state_1))
#
#             state_0 = state_1
#     return Data(states, acts, next_states)


# def yaml_to_namespace(yaml_file):
#     with open(yaml_file, 'r') as file:
#         config = yaml.safe_load(file)
#         return convert_to_namespace(config)
#
#
# def convert_to_namespace(d):
#     if isinstance(d, dict):
#         return SimpleNamespace(**{k: convert_to_namespace(v) for k, v in d.items()})
#     elif isinstance(d, list):
#         return [convert_to_namespace(i) for i in d]
#     else:
#         return d