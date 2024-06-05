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

#import the utils libraries
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path,'/src/utils')
sys.path.append(module_path)
from utils import TrajectoryHandler, load_trajectories, process_obs

agent_type = 'stable'

class ImitationNode:
    def __init__(self,rec=False):
        rospy.init_node("imitation_learning_commands", anonymous=True)
        self.names_right = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
        self.load_dir = os.path.join(package_path,"agent_params")
        self.data_dir = os.path.join(package_path,"data")
        task = rospy.get_param('task')
        self.recording = rospy.get_param('rec')

        #ROS stuff
        rospy.Subscriber('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes)
        self.publisher_joint_commands = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        self.publisher_ee_commands = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right)
        self.tf_listener = tf.TransformListener()
        rate = 1
        self.control_loop_rate = rospy.Rate(rate)  # 10Hz
        rospy.sleep(2)

        self.block=False


        self.curr_joints_pos = np.array([])
        self.curr_joints_vel = np.array([])
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.goal = np.expand_dims(np.array([ 4.08305303e-01, -1.07297375e-05,  1.27501961e+00]),0)
        self.pos = np.array([0.093, 0.1, -0.0728])
        self.n_frames = 4
        self.k_v = 20

        #initialize the agent

        if agent_type=='stable':
            from utils_rl import AgentStable
            self.agent = AgentStable(input_size=59, hidden_size1=128, hidden_size2=128, encoder_hidden_size = 64, output_size=3, 
                            device='cuda' if torch.cuda.is_available() else 'cpu')
            self.agent.encoder.load_model(os.path.join(self.load_dir,f'model_encode.pth'))
            self.agent.load_model(os.path.join(self.load_dir,f'model_{task}_stable.pth'))
            # get all trajectories in equivariant representation and create the density estimator 
            dataset = load_trajectories(
                data_path=os.path.join(self.data_dir, task),
                n_frames=self.n_frames,
                n_freq = int(50/rate)
            )
            self.agent.init_KDE(dataset.states)
        else:
            from utils_rl import Agent
            self.agent = Agent(input_size=59, hidden_size1=128, hidden_size2=128, output_size=3, 
                            device='cuda' if torch.cuda.is_available() else 'cpu')
            self.agent.load_model(os.path.join(self.load_dir,f'model_{task}.pth'))
            self.agent.to_device()

        if self.recording:
            self.traj_data = TrajectoryHandler(save_dir=self.data_dir)
        rospy.sleep(2)


    def main(self):
        
        # init the observation stack
        obs_dict = {
            'goal': self.goal,
            'joint_pos': deque(list(self.curr_joints_pos)*self.n_frames,maxlen=self.n_frames),
            'joint_vel': deque(list(self.curr_joints_vel)*self.n_frames,maxlen=self.n_frames),
        }
        rospy.sleep(2)    

        print(f"Starting ... aiming to {self.goal}")
        while not rospy.is_shutdown():
            # process the observations and collect the acytion from the agent
            obs_dict['joint_pos'].append(self.curr_joints_pos[0])
            obs_dict['joint_vel'].append(self.curr_joints_vel[0])
            obs_tsr = process_obs(obs_dict).to(self.agent.device)
            action = self.agent.select_action(obs_tsr).detach().cpu().squeeze().numpy()     
            
            rec_pos_,rec_or_ = self.get_transform(target_frame='base_footprint',source_frame=f'inner_finger_1_right_link')
            

            # publish a new commanded pos
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
                        
            self.control_loop_rate.sleep()


    def callback_joint_state(self,joints_msg):
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)

    def simulator_remove_grape_bunch(self,id_: int):
        rospy.wait_for_service('/simulator')
        cmd = rospy.ServiceProxy('/simulator', Simulator)
        cmd("RemoveGrapeBunch", id_, False, "")
        print(f'{id_} grape removed')

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
                self.traj_data.save_trajectory('imitation_stable')


    def get_transform(self, target_frame, source_frame):
        try:
            # Wait for the transform to become available and get the transform
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return None, None

    def joint_PD(self,target_pos):
        joint_state = rospy.wait_for_message('canopies_simulator/joint_states', JointState, 10)
        joint_vel_command = [0]*7
        error = 0
        for name in self.names_right:
            index_in_joint_state = joint_state.name.index(name)
            index_in_msg = self.names_right.index(name)
            final_pos = target_pos[index_in_msg]
            real_pos = joint_state.position[index_in_joint_state]
            e_i = final_pos - real_pos
            control_velocity = 10 * e_i
            error += e_i**2
            joint_vel_command[index_in_msg] = control_velocity
        error = np.sqrt(error)
        return joint_vel_command, error



def update_stack(obs_stack, goal, j_pos, j_vel):
    obs_stack['goal'] = goal

    return obs_stack



if __name__ == "__main__":
    try:
        node = ImitationNode(rec=True)
        node.main()
    except rospy.ROSInterruptException:
        pass






