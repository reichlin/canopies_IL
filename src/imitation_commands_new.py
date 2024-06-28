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

#import the utils libraries
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path,'/src/utils')
sys.path.append(module_path)
from utils import TrajectoryHandler, load_trajectories, process_obs
from utils_rl import Agent

random_init = False

# 232 228 97 160


class ImitationNode:
    def __init__(self,rec=False):
        rospy.init_node("high_level_controller", anonymous=True)
        self.block = False
        self.activate_pd = True

        #init global vars
        self.bunch_poses = {0:np.array([0.5, -0.2, 0.5])}
        self.target_ids = [97, 160]
        self.init_pos = np.array([0.4081436362262652, -0.43626031658396147, 0.537828329572042])
        
        #get rosparams
        self.names_right = rospy.get_param('/arm_right_joints') 
        self.load_dir = os.path.join(package_path,rospy.get_param('/folders/load'))
        self.data_dir = os.path.join(package_path,rospy.get_param('/folders/data'))
        self.recording = rospy.get_param('rec')
        self.recording = False   #TODO:   cancelllllllllllllllllllllllllllllllllllllllllllllllllll
        self.n_frames = rospy.get_param('/agent_hp/n_frames')

        task = rospy.get_param('task')
        self.saving_name = rospy.get_param('agent_name')
        self.stable_agent = True if self.saving_name=='stable_agent' else False

        #ROS stuff
        rospy.Subscriber('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes)
        self.publisher_joint_commands = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        self.publisher_ee_commands = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right)
        self.tf_listener = tf.TransformListener()
        rate=rospy.get_param('rates/testing')
        self.control_loop_rate = rospy.Rate(1)  # 1Hz
        rospy.sleep(2)


        joints_msg = rospy.wait_for_message('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, timeout=None)
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0]*7
        self.pos = np.zeros(3) 
        self.k_v = rospy.get_param('/gains/k_v')


        #initialize the agent
        input_dim = rospy.get_param('/agent_hp/input_dim') 
        output_dim = rospy.get_param('/agent_hp/output_dim')
        hidden_dim = rospy.get_param('/agent_hp/hidden_dim') 
        self.agent = Agent(input_size=input_dim, hidden_size1=hidden_dim, hidden_size2=hidden_dim, output_size=output_dim, 
                            stable=self.stable_agent, device='cuda' if torch.cuda.is_available() else 'cpu')
        model2load = f'model_{task}_stable.pth' if self.stable_agent else f'model_{task}.pth'
        self.agent.load_model(os.path.join(self.load_dir,model2load))
        if self.stable_agent:
            #need to load the params for the encoder and to init the KDE
            self.agent.encoder.load_model(os.path.join(self.load_dir,f'model_encode.pth'))
            n_freq = int(rospy.get_param('rates/recording')/rate)
            dataset = load_trajectories(
                            data_path=os.path.join(self.data_dir, task),
                            n_frames=self.n_frames,
                            n_freq = n_freq
                        )
            z = self.agent.init_KDE(dataset.states)
        self.agent.to_device()
        
        #setup the recorder
        if self.recording:
            path = os.path.join(self.data_dir,f'{task}_results')
            self.traj_data = TrajectoryHandler(save_dir=path, tag=task)
        rospy.sleep(2)


    def main(self):

        print(f"Starting with target grape bunches {self.target_ids}")

        self.goal_id = self.target_ids.pop(0)
        self.goal = self.bunch_poses[self.goal_id]
        
        self.pd_traj = self.generate_parabolic_trajectory(self.goal)
        self.goal_pd = self.pd_traj[0]

        '''print(list(self.bunch_poses.keys()))
        for g in self.bunch_poses.keys():
            input(g)
            self.simulator_remove_grape_bunch(g)'''

        # init the observation stack
        obs_dict = {
            'goal': np.expand_dims(self.goal,0),
            'joint_pos': deque(list(self.curr_joints_pos)*self.n_frames,maxlen=self.n_frames),
            'joint_vel': deque(list(self.curr_joints_vel)*self.n_frames,maxlen=self.n_frames),
        }
        rospy.sleep(2)    
                
        sim_step=0
        if random_init:
            self.pos = self.pos + np.concatenate([np.random.uniform(low=-0.3, high=0.3, size=1), 
                            np.random.uniform(low=0, high=0.3, size=1),
                            np.random.uniform(low=0, high=0.3, size=1)])

        while not rospy.is_shutdown():
            rec_pos_,rec_or_ = self.get_transform(target_frame='base_footprint',source_frame=f'inner_finger_1_right_link')

            if self.block:
                #print('Blocking')
                self.velocity_msg_right.data = [0.0]*7
                self.publisher_joint_commands.publish(self.velocity_msg_right)
                self.control_loop_rate.sleep()
                continue

            elif self.activate_pd:
                self.check_traj_tracking(rec_pos_)
                action = self.cartesian_PD(self.goal_pd, np.array(rec_pos_))    

            else:
                obs_dict['joint_pos'].append(self.curr_joints_pos[0])
                obs_dict['joint_vel'].append(self.curr_joints_vel[0])
                obs_dict['goal'] = np.expand_dims(self.goal,0)

                obs_tsr = process_obs(obs_dict).to(self.agent.device)
                action = self.agent.select_action(obs_tsr).detach().cpu().squeeze().numpy() 
            
            # publish a new commanded pos
            if not(random_init and sim_step<10):
                self.pos += action

            t = time.time()
            ee_pos_msg = ExternalReference()
            ee_pos_msg.position = Point(x=self.pos[0], y=self.pos[1], z=self.pos[2])
            ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            self.publisher_ee_commands.publish(ee_pos_msg)

            if self.recording:
                rec_joints_pos = copy.deepcopy(self.curr_joints_pos)
                rec_joints_vel = copy.deepcopy(self.curr_joints_vel)
                rec_pos_,rec_or_ = self.get_transform(target_frame='base_footprint',source_frame=f'inner_finger_1_right_link')
                rec_pos = np.expand_dims(np.array(rec_pos_), 0)
                rec_or = np.expand_dims(np.array(rec_or_), 0)
            
            #wait for the velocity command 
            joint_vel_ik_right = rospy.wait_for_message('/arm_right_forward_velocity_controller/command',Float64MultiArray, timeout=None)

            
            for i, name in enumerate(self.names_right):
                index_in_msg = self.names_right.index(name)
                self.velocity_msg_right.data[index_in_msg] = joint_vel_ik_right.data[i] * self.k_v
            
            self.publisher_joint_commands.publish(self.velocity_msg_right)

            #rec variables
            if self.recording>0.0:
                    rec_joint_action = copy.deepcopy(np.expand_dims(np.array(self.velocity_msg_right.data), 0))
                    self.traj_data.store_joints_pos(rec_joints_pos)
                    self.traj_data.store_joints_vel(rec_joints_vel)
                    self.traj_data.store_pos(rec_pos)
                    self.traj_data.store_orientation(rec_or)
                    self.traj_data.store_velocity(rec_joint_action)
                    self.traj_data.store_action(np.expand_dims(action,0))
            sim_step+=1

            self.control_loop_rate.sleep()

    def cartesian_PD(self, g, p):
        gay = np.array([0.1]*3)
        dp = g-p
        a = np.clip(dp*gay,-0.1,0.1)
        return a


    def callback_joint_state(self,joints_msg):
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)

    def simulator_remove_grape_bunch(self,id_: int):
        rospy.wait_for_service('/simulator')
        cmd = rospy.ServiceProxy('/simulator', Simulator)
        cmd("RemoveGrapeBunch", id_, False, "")
        print(f'Grape bunch {id_} removed')

    def callback_grapes(self, grapes_data):
        ee_pos_1,_ = self.get_transform(target_frame='base_footprint',source_frame=f'fingertip_1_right_link')
        ee_pos_2,_ = self.get_transform(target_frame='base_footprint',source_frame=f'fingertip_2_right_link')
        ee_pos = (np.array(ee_pos_1) + np.array(ee_pos_2))/2 
        for box in grapes_data.boxes:
            i = box.index
            g_pos, _ = self.get_transform(target_frame='base_footprint',source_frame=f'Bunch_{i}')
            dist = np.linalg.norm(np.array(ee_pos)-np.array(g_pos))
            if dist<0.2:
                self.simulator_remove_grape_bunch(int(box.index))
                self.update_goal(i)
                #if self.recording: self.traj_data.save_trajectory(self.saving_name)

            #update bounches to be seen
            self.bunch_poses[i] = np.array(g_pos)


    def get_transform(self, target_frame, source_frame):
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return None, None

    def update_goal(self, idx):
        goal_reached = idx == self.goal_id
        if goal_reached:
            if self.target_ids == []:
                print(f'Last goal reached!')
                self.block = True
            else:
                self.goal_id = self.target_ids.pop(0)
                self.goal = self.bunch_poses[self.goal_id]
                print(f'goal updated to  {self.goal_id}')
                self.pd_traj = self.generate_parabolic_trajectory(self.goal)
                self.goal_pd = self.pd_traj[0]
                self.activate_pd = True
                self.block = True
                time.sleep(1)
                self.block=False


    def check_traj_tracking(self, pos):
        e = np.linalg.norm(pos-self.goal_pd)
        if e<0.1:
            self.goal_pd = self.pd_traj.pop()
        else:
            self.goal_pd = self.pd_traj[0]
    
    '''
    def generate_parabolic_trajectory(self, end, num_points=10):

        start,_ = self.get_transform(target_frame='base_footprint',source_frame=f'inner_finger_1_right_link')

        # Unpack the start, end, and peak points
        x1, y1, z1 = start
        x2, y2, z2 = end
        xp, yp, zp = self.init_pos
        
        # Generate linearly spaced t values
        t = np.linspace(0, 1, num_points)
        
        # Parametric equation for x and y (quadratic interpolation)
        x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * xp + t ** 2 * x2
        y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * yp + t ** 2 * y2
        
        # Parametric equation for z (quadratic interpolation)
        z = (1 - t) ** 2 * z1 + 2 * (1 - t) * t * zp + t ** 2 * z2
        
        # Combine into a single array of points
        trajectory = np.vstack((x, y, z)).T
        return list(trajectory)

    '''
    def generate_parabolic_trajectory(self, end, num_points=100):

        start,_ = self.get_transform(target_frame='base_footprint',source_frame=f'inner_finger_1_right_link')
        # Unpack the start and end points
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        # Generate linearly spaced x and y points
        t = np.linspace(0, 1, num_points)
        
        # Parametric equation for x and y (linear interpolation)
        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2
        
        # Parabolic interpolation for z
        z = (1 - t) * z1 + t * z2 - 4 * t * (1 - t)  # Parabolic shape factor
        
        # Combine into a single array of points
        trajectory = np.vstack((x, y, z)).T
        return list(trajectory)

    


if __name__ == "__main__":
    try:
        node = ImitationNode(rec=True)
        node.main()
    except rospy.ROSInterruptException:
        pass






