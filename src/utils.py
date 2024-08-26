import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from collections import deque, namedtuple
import rospy
import pickle
from canopies_simulator.srv import Simulator


def check_workspace(pos:np.ndarray, c:np.ndarray, h:float, r_bounds:tuple, theta_bounds:tuple, idx:str=None):
    p = pos - c
    dist_xy = np.linalg.norm(p[:2])
    dist_z = np.linalg.norm(p[2])
    dist_th = np.arctan2(p[1], p[0])
    return (dist_xy <= r_bounds[1] and
            dist_xy >= r_bounds[0] and
            dist_z <= h and
            dist_th >= theta_bounds[0] and
            dist_th <= theta_bounds[1])

def simulator_remove_grape_bunch(id_: int):
    """
    Removes the bunch corresponding to the input id
    """
    rospy.wait_for_service('/simulator')
    cmd = rospy.ServiceProxy('/simulator', Simulator)
    cmd("RemoveGrapeBunch", id_, False, "")

def get_closest_obj(p:np.ndarray, poses:np.ndarray, ids:list):
    '''
    INPUT:
        - p: RF position
        - poses: array of all object positions
        - ids: list of object ids
    OUTPUT: id and distance of the closest object
    '''
    dists_3d = poses - np.array(p)
    dists = np.linalg.norm(dists_3d, axis=-1)
    idx = np.argmin(dists)
    return ids[idx], dists_3d[idx]


def communicate_instructions(target_id,target_dist):
    dx, dy, dz = target_dist
    rospy.loginfo(f'\n{target_id}: {target_dist} ({np.linalg.norm(target_dist)}):')
    rospy.loginfo(f' - X: {"Forward" if dx >0.0 else "Backward"} by {abs(dx)}')
    rospy.loginfo(f' - Y: {"Left" if dx >0. else "Right"} by {abs(dy)}')
    rospy.loginfo(f' - Z: {"Up" if dx >0. else "Down"} by {abs(dz)}')



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
        self.target_positions = dict(value=[], time=[])
        self.target_orientations = dict(value=[], time=[])
        self.command_positions = dict(value=[], time=[])
        self.grapes_positions = []

    def size(self):
        return len(self.cartesian_positions['value'])

    def save_trajectory(self, name):
        d = dict(
            joint_positions=self.joint_positions,
            joint_velocities=self.joint_velocities,
            cartesian_positions=self.cartesian_positions,
            cartesian_orientations=self.cartesian_orientations,
            target_positions=self.target_positions,
            target_orientations=self.target_orientations,
            command_positions=self.command_positions,
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
        rospy.loginfo(f'vr dpositions: {len(self.command_positions["value"])}')
        rospy.loginfo(f'------------------------------------')