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


names_right = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
names_left = ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint"]
block = 0.0
load_dir = "/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/agent_params"
grapes_info = np.array([])
curr_joints_pos = np.array([])
curr_joints_vel = np.array([])

goal = [4.765594005584717, -1.0406124591827393, 1.8432689905166626]

def callback_end_effector_state(end_effector_msg):
    pos = end_effector_msg.pose.position
    orient = end_effector_msg.pose.orientation
    error = np.array(goal)- np.array([pos.x,pos.y,pos.z] )
    #print(f'{error} --> {np.linalg.norm(error)}')
    #curr_or = np.expand_dims(np.array([orient.x,orient.y,orient.z, orient.w]), 0)

def callback_grapes_info(grapes_data):
    global grapes_info
    grapes_info = grapes_data

def callback_joint_state(joints_msg):
    global curr_joints_pos, curr_joints_vel
    curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
    curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)

def store_external_objs(self, objects:list):
    data_external_objs = []
    for box in objects:
        data_external_objs.append(np.array([[[box.xmin, box.xmax], [box.ymin, box.ymax], [box.zmin, box.zmax]]]))
    data_external_objs = np.concatenate(data_external_objs, 0)
    #print('data_external_objs shape: ',self.data_external_objs.shape)


def callback_grapes_info(graps_pos):
    data_external_objs = []
    for box in graps_pos.boxes:
        data_external_objs.append(np.array([[[box.xmin, box.xmax], [box.ymin, box.ymax], [box.zmin, box.zmax]]]))
    data_external_objs = np.concatenate(data_external_objs, 0)


def main():

    rospy.init_node("imitation_learning_commands", anonymous=True)

    publisher = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)

    rospy.Subscriber('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, callback_joint_state)
    #rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray,callback_grapes_info)
    rospy.Subscriber('/task_2_value', PoseStamped, callback_end_effector_state)

    #rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, callback_vr_inputs_right_arm)

    rospy.sleep(2)

    control_loop_rate = rospy.Rate(10)  # 10Hz

    names_right = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
    names_left = ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint"]
    rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', names_right)

    #initialize the agent
    task = rospy.get_param('task')
    agent = Agent(input_size=17, hidden_size1=64, hidden_size2=64, output_size=7, 
                    device='cuda' if torch.cuda.is_available() else 'cpu')
    print(agent.device,'--------------')

    agent.load_model(os.path.join(load_dir,f'model_{task}.pth'))
    agent.to(agent.device)
    rospy.sleep(2)
    

    #TODO: init action

    print(f"Starting ... aiming at {goal}")
    cnt = 0
    while not rospy.is_shutdown():
        target_pos = np.expand_dims(np.array(goal), 0)

        obs_tsr = torch.from_numpy(np.concatenate((target_pos, curr_joints_pos, curr_joints_vel), -1)).float().to(agent.device)
        action = agent(obs_tsr).detach().cpu().squeeze().numpy()      
            
        action_msg = Float64MultiArray()
        action_msg.data = [a*1 for a in action]  
        publisher.publish(action_msg)
        control_loop_rate.sleep()

    action_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    publisher.publish(velocity_msg_right)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Agent(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, device='cpu'):
        super(Agent, self).__init__()
        self.model = MLP(input_size, hidden_size1, hidden_size2, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device
    
    def forward(self,x):
        return self.model(x)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model parameters loaded from {path}")

    def process_obs(self, target_pos, joint_pos, joint_vel):
        #find the closest obj
        #obs = np.concatenate([target_pos, joint_pos, joint_vel],0)
        obs_tsr = torch.tensor([target_pos, joint_pos, joint_vel], dtype=torch.float32)
        return obs_tsr





if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass






