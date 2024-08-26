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
from utils import TrajectoryHandler, process_obs
from utils_rl import Agent
import datetime



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


        #init variables
        self.curr_joints_pos, self.curr_joints_vel = np.zeros((1, 7)), np.zeros((1, 7))
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0] * 8
        self.goal = None
        self.pos = np.zeros(3)
        self.k_v = rospy.get_param('/gains/k_v')

        # ROS stuff
        rospy.Subscriber('/canopies_simulator/joint_states', JointState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes, queue_size=1)
        self.publisher_joint_commands = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        self.publisher_ee_commands = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right+ ['torso_lift_joint'])
        rospy.set_param('canopies_simulator/joint_states/rate', 100)
        self.tf_listener = tf.TransformListener()
        rate = rospy.get_param('rates/testing')
        self.control_loop_rate = rospy.Rate(rate)
        rospy.sleep(2)

        # init agent and load param
        self.task = rospy.get_param('task')
        self.stable = rospy.get_param('stable')
        self.agent = Agent(
            input_size=rospy.get_param('/agent_hp/input_dim')*self.n_frames,
            goal_size=3,
            action_size=rospy.get_param('/agent_hp/output_dim'), 
            N_gaussians=25,
            stable= self.stable, # if rospy.get_param('agent_name') else False,
            device=self.device
        ).to(self.device)

        model_file = os.path.join(package_path, self.load_dir, f"model_{self.task}.pth" )

        self.agent.load_model(model_file)

        # setup the recorder
        if self.recording:
            self.traj_data = TrajectoryHandler(save_dir=self.save_dir)

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

        # get closest grape to be the goal
        '''g_mean = np.array([[ 0.3065, -0.2552,  1.5596]]) #np.array([[0.5, -0.3, 1.3]])
        closest_grape_idx = np.argmin(np.linalg.norm(self.grapes_pos - g_mean, axis=-1))
        self.goal = torch.from_numpy(self.grapes_pos[closest_grape_idx:closest_grape_idx + 1]).float().to(self.agent.device)
        goal_id = self.grapes_idx[closest_grape_idx]
        print(goal_id)
        '''

        #init current variables and observation stack   
        joints_msg = rospy.wait_for_message('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, timeout=None)
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)
        obs_dict = {
            'joint_pos': deque(list(self.curr_joints_pos) * self.n_frames, maxlen=self.n_frames),
            'joint_vel': deque(list(self.curr_joints_vel) * self.n_frames, maxlen=self.n_frames),
        }
        rospy.sleep(2)
        
        # give a random initial position if necessaire
        '''
        if random_init:
            self.pos = self.pos + np.concatenate([np.random.uniform(low=-0.3, high=0.3, size=1),
                                                  np.random.uniform(low=0, high=0.3, size=1),
                                                  np.random.uniform(low=0, high=0.3, size=1)])
        '''

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

            # publish a new commanded pos
            if not (random_init and sim_step < 10):
                self.pos += action
            ee_pos_msg.position = Point(x=self.pos[0], y=self.pos[1], z=self.pos[2])
            ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            self.publisher_ee_commands.publish(ee_pos_msg)

            #remove a grape if necessary
            self.grape_revove_check(rec_pos)

            if self.recording:
                rec_joints_pos = copy.deepcopy(self.curr_joints_pos)
                rec_joints_vel = copy.deepcopy(self.curr_joints_vel)
                rec_pos = np.expand_dims(np.array(rec_pos), 0)
                rec_or = np.expand_dims(np.array(rec_or), 0)

            # wait for the velocity command and forward to the simulator
            joint_vel_ik_right = rospy.wait_for_message('/arm_right_forward_velocity_controller/command', Float64MultiArray, timeout=None)
            for i, name in enumerate(self.names_right):
                index_in_msg = self.names_right.index(name)
                self.velocity_msg_right.data[index_in_msg] = joint_vel_ik_right.data[i] * self.k_v
            self.publisher_joint_commands.publish(self.velocity_msg_right)

            # rec variables
            if self.recording:
                self.traj_data.store_data(
                    (
                        rec_joints_pos,
                        rec_joints_vel,
                        rec_pos,
                        rec_or,
                        copy.deepcopy(np.expand_dims(np.array(self.velocity_msg_right.data), 0)),
                        None,
                        self.grapes_pos
                    )
                )

                if sim_step==1000 or self.grapes_idx in self.removed_grapes:
                    self.traj_data.save_trajectory(f'{self.task}{"_stable" if self.stable else ""}')


            sim_step += 1
            self.control_loop_rate.sleep()



    ## ----------------- CALLBACKS -----------------

    def callback_joint_state(self, joints_msg):
        for name in self.names_right:
            index_in_joint_state = joints_msg.name.index(name)
            index_in_msg = self.names_right.index(name)
            self.curr_joints_pos[0, index_in_msg] = joints_msg.position[index_in_joint_state]
            self.curr_joints_vel[0, index_in_msg] = joints_msg.velocity[index_in_joint_state]
    
    def callback_grapes(self, grapes_data):
        grapes_pos, grapes_idx = [], []
        for box in grapes_data.boxes:
            grapes_idx.append(box.index)
            g_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{box.index}')
            grapes_pos.append(g_pos)
        self.grapes_pos = np.array(grapes_pos)
        self.grapes_idx = grapes_idx


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


if __name__ == "__main__":
    try:
        node = ImitationNode(rec=True)
        node.main()
    except rospy.ROSInterruptException:
        pass
