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
                     vr_act=np.concatenate(self.data_vr_act, 0) if len(self.data_vr_act) > 0 else np.zeros(1),
                     )
            print(f"Trajectory saved at {file}")
            self.show_current_status()
            return file
        else:
            return 'NEVER'




import pickle

class Buffer:
    def __init__(self, save_dir):
        self.reset()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def reset(self):
        self.joint_positions = dict(value=[], time=[])
        self.joint_velocities = dict(value=[], time=[])
        self.cartesian_positions = dict(value=[], time=[])
        self.cartesian_orientations = dict(value=[], time=[])
        self.vr_commands = dict(value=[], time=[])
        self.grapes_positions = []

    def size(self):
        return len(self.cartesian_positions['value'])

    def save_trajectory(self, name):
        d = dict(
            joint_positions=self.joint_positions,
            joint_velocities=self.joint_velocities,
            cartesian_positions=self.cartesian_positions,
            cartesian_orientations=self.cartesian_orientations,
            vr_commands=self.vr_commands,
            grapes_positions=self.grapes_positions
        )
        file = os.path.join(self.save_dir, name)

        with open(file, 'wb') as f:
            pickle.dump(d, f)
        return file

    def show_current_status(self):
        rospy.loginfo(f'------------- SUMMARY -------------')
        rospy.loginfo(f'joint positions: {len(self.joint_positions["value"])}')
        rospy.loginfo(f'joint velocities: {len(self.joint_velocities["value"])}')
        rospy.loginfo(f'ee positions: {len(self.cartesian_positions["value"])}')
        rospy.loginfo(f'ee orientations: {len(self.cartesian_orientations["value"])}')
        rospy.loginfo(f'vr commands: {len(self.vr_commands["value"])}')
        rospy.loginfo(f'------------------------------------')