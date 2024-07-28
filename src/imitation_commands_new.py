#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped, TwistStamped, Twist
from std_msgs.msg import Float64MultiArray
from farming_robot_control_msgs.msg import ExternalReference
from quest2ros.msg import OVR2ROSInputs
from control_msgs.msg import JointTrajectoryControllerState
from canopies_simulator.msg import BoundBoxArray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import copy
import time
from canopies_simulator.srv import Simulator
import tf
import matplotlib.pyplot as plt

# import the utils libraries
import rospkg

rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path, '/src/utils')
sys.path.append(module_path)
from utils import Buffer
from utils_rl import Agent



random_init = False


class ImitationNode:

    def __init__(self, rec=False):

        rospy.init_node("high_level_controller", anonymous=True)

        # get rosparams
        self.names_right = rospy.get_param('/arm_right_joints')
        self.load_dir = os.path.join(package_path, rospy.get_param('/folders/load'))
        self.data_dir = os.path.join(package_path, rospy.get_param('/folders/data'))
        self.save_dir = os.path.join(package_path, rospy.get_param('/folders/result'))

        self.recording = rospy.get_param('rec')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.threshold = rospy.get_param('/threshold')
        self.n_frames = rospy.get_param('/agent_hp/n_frames')

#[0.64730491 0.00703621 1.25233966]
        #init variables
        self.curr_joints_pos, self.curr_joints_vel = np.zeros((1, 7)), np.zeros((1, 7))
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0] * 8
        self.goal = None
        self.pos = np.zeros(3)
        self.k_v = rospy.get_param('/gains/k_v')

        # setup the recorder
        if self.recording:
            self.traj_buffer = Buffer(self.save_dir)

        # ROS stuff
        rospy.Subscriber('/canopies_simulator/joint_states', JointState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes, queue_size=1)
        self.publisher_joint_commands = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        self.publisher_ee_commands = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        rospy.Subscriber('/direct_kinematics/rigth_arm_end_effector', PoseStamped, self.callback_end_effector, queue_size=1)

        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right+ ['torso_lift_joint'])
        rospy.set_param('canopies_simulator/joint_states/rate', 100)
        self.tf_listener = tf.TransformListener()
        rate = rospy.get_param('rates/testing')
        self.control_loop_rate = rospy.Rate(rate)
        self.rate=rate
        rospy.sleep(2)

        # init agent and load param
        self.task = rospy.get_param('task')
        self.stable = rospy.get_param('stable')
        self.agent = Agent(
            input_size= 7, #rospy.get_param('/agent_hp/input_dim')*self.n_frames,
            goal_size=3,
            action_size=rospy.get_param('/agent_hp/output_dim'), 
            N_gaussians=25,
            stable=self.stable, # if rospy.get_param('agent_name') else False,
            device=self.device
        ).to(self.device)

        model_file = os.path.join(package_path, self.load_dir, f"model_{self.task}_50.pth" )

        self.agent.load_model(model_file)



        # init the grapes
        self.grapes_pos, self.grapes_idx, self.removed_grapes = self.register_grapes()

        # lift torso
        q_torso = rospy.get_param('torso_height')
        velocity_msg_right = Float64MultiArray()
        velocity_msg_right.data = [0.0] * 7 + [q_torso]
        self.publisher_joint_commands.publish(velocity_msg_right)
        rospy.sleep(1)
        velocity_msg_right.data = [0.0] * 8
        self.publisher_joint_commands.publish(velocity_msg_right)
        rospy.sleep(2)


    def main(self):


        input(f"\n{'-'*50}\nLocate your self in the simulator. Once done, press enter to continue.")

        available_ids = {}
        print("\nThe grapes reachable from this position are the following:")
        for g_id, g_pos in zip(self.grapes_idx, self.grapes_pos):
            dist = np.linalg.norm(g_pos)
            if dist<1.5:
                print(f' - {g_id} at distance {round(dist, 3)}m from the basefootprint')
                available_ids[g_id] = g_pos

        goal_id = int(input("Input the id of the grape to collect?: "))
        while not goal_id in available_ids:
            goal_id = int(input("id not valid. Please enter a valid id (from the list): "))

        goal_pos = available_ids[goal_id]
        self.goal = torch.tensor(goal_pos).float().unsqueeze(0).to(self.device)
        ## ----------------- SETUP -----------------

        #init current variables and observation stack   
        joints_msg = rospy.wait_for_message('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, timeout=None)
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)
        obs_dict = {
            'joint_pos': deque(list(self.curr_joints_pos) * self.n_frames, maxlen=self.n_frames),
            'joint_vel': deque(list(self.curr_joints_vel) * self.n_frames, maxlen=self.n_frames),
        }
        rospy.sleep(2)

        ## ----------------- SIM LOOP -----------------

        rospy.loginfo(f"Starting ... agent {'stable' if self.stable else ''} aiming to {goal_pos} of id {goal_id}")
        sim_step = 0
        ee_pos_msg = ExternalReference()

        while not rospy.is_shutdown():

            # process the observations and collect the action from the agent
            obs_dict['joint_pos'].append(self.curr_joints_pos[0])
            obs_dict['joint_vel'].append(self.curr_joints_vel[0])
            obs_tsr = process_obs(obs_dict).to(self.agent.device)
            action = self.agent.select_action(obs_tsr, self.goal).detach().cpu().squeeze().numpy()
            rec_pos, rec_or = self.get_transform(target_frame='base_footprint', source_frame=f'inner_finger_1_right_link')
            if self.recording:
                self.traj_buffer.vr_commands['value'].append(list(action))
                self.traj_buffer.vr_commands['time'].append(rospy.get_time())

            # publish a new commanded pos
            if not (random_init and sim_step < 10):
                self.pos += action
            ee_pos_msg.position = Point(x=self.pos[0], y=self.pos[1], z=self.pos[2])
            ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            self.publisher_ee_commands.publish(ee_pos_msg)

            #remove a grape if necessary
            self.grape_revove_check(rec_pos)

            # wait for the velocity command and forward to the simulator
            joint_vel_ik_right = rospy.wait_for_message('/arm_right_forward_velocity_controller/command', Float64MultiArray, timeout=None)
            for i, name in enumerate(self.names_right):
                index_in_msg = self.names_right.index(name)
                self.velocity_msg_right.data[index_in_msg] = joint_vel_ik_right.data[i] * self.k_v
            self.publisher_joint_commands.publish(self.velocity_msg_right)

            # rec variables
            if self.recording and (goal_id in self.removed_grapes or sim_step == 1000):
                self.traj_buffer.grapes_positions.append([self.goal])
                self.traj_buffer.show_current_status()
                self.traj_buffer.save_trajectory(f'{self.task}{"_stable" if self.stable else ""}_{self.rate}.pkl')
            sim_step += 1
            self.control_loop_rate.sleep()



    ## ----------------- CALLBACKS -----------------

    def callback_joint_state(self, joints_msg):
        t = rospy.get_time()
        joints_pos = [0.] * len(self.names_right)
        joints_vel = [0.] * len(self.names_right)
        for name in self.names_right:
            index_in_joint_state = joints_msg.name.index(name)
            index_in_msg = self.names_right.index(name)
            joints_pos[index_in_msg] = joints_msg.position[index_in_joint_state]
            joints_vel[index_in_msg] = joints_msg.velocity[index_in_joint_state]

        if self.recording:
            self.traj_buffer.joint_positions['value'].append(joints_pos)
            self.traj_buffer.joint_velocities['value'].append(joints_vel)
            self.traj_buffer.joint_positions['time'].append(t)
            self.traj_buffer.joint_velocities['time'].append(t)


    def callback_grapes(self, grapes_data):
        grapes_pos, grapes_idx = [], []
        for box in grapes_data.boxes:
            grapes_idx.append(box.index)
            g_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{box.index}')
            grapes_pos.append(g_pos)
        self.grapes_pos = np.array(grapes_pos)
        self.grapes_idx = grapes_idx

    def callback_end_effector(self, pose_msg):
        t = pose_msg.header.stamp.secs + pose_msg.header.stamp.nsecs/(10**9)
        self.ee_pos = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        self.ee_rot = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]
        if self.recording:
            self.traj_buffer.cartesian_positions['value'].append(self.ee_pos)
            self.traj_buffer.cartesian_orientations['value'].append(self.ee_rot)
            self.traj_buffer.cartesian_positions['time'].append(t)
            self.traj_buffer.cartesian_orientations['time'].append(t)


    ## ----------------- METHODS -----------------

    def simulator_remove_grape_bunch(self, id_: int):
        rospy.wait_for_service('/simulator')
        cmd = rospy.ServiceProxy('/simulator', Simulator)
        cmd("RemoveGrapeBunch", id_, False, "")


    def grape_revove_check(self, ee_pos):
        dists = np.linalg.norm(np.array(ee_pos) - self.grapes_pos, axis=-1)
        threshold = (dists < self.threshold)*1.
        for i in np.nonzero(threshold)[0]:
            if self.grapes_idx[i] not in self.removed_grapes:
                self.simulator_remove_grape_bunch(int(self.grapes_idx[i]))
                rospy.loginfo(f'\n{self.grapes_idx[i]} grape removed')
                self.removed_grapes.append(self.grapes_idx[i])


    def get_transform(self, target_frame, source_frame):
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return None, None

    def register_grapes(self):
        grapes_pos, grapes_idx = [], []
        grapes_data = rospy.wait_for_message('/canopies_simulator/grape_boxes', BoundBoxArray, timeout=None)
        for box in grapes_data.boxes:
            grapes_idx.append(box.index)
            g_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{box.index}')
            grapes_pos.append(g_pos)
        grapes_pos = np.array(grapes_pos)
        return grapes_pos, grapes_idx, []


def process_obs(obs_dict):
    joint_pos = torch.from_numpy(np.expand_dims(np.stack(obs_dict['joint_pos'], axis=0).flatten(), 0)).float()
    joint_vel = torch.from_numpy(np.expand_dims(np.stack(obs_dict['joint_vel'], axis=0).flatten(), 0)).float()
    obs_tsr = torch.cat([joint_pos, joint_vel], dim=1)
    return joint_pos #obs_tsr



if __name__ == "__main__":
    try:
        node = ImitationNode(rec=True)
        node.main()
    except rospy.ROSInterruptException:
        pass
