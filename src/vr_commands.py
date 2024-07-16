#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped, TwistStamped, Twist
from std_msgs.msg import Float64MultiArray
from farming_robot_control_msgs.msg import ExternalReference
from canopies_simulator.msg import BoundBoxArray
from quest2ros.msg import OVR2ROSInputs
import os
import numpy as np
import time
import datetime
import copy
import rospkg
import tf
from scipy.spatial.transform import Rotation as R
from canopies_simulator.srv import Simulator
#import the utils libraries
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path,'/src/utils')
sys.path.append(module_path)
from utils import TrajectoryHandler


class VRCommands:
    def __init__(self): 
        
        #status variables
        self.resetting = False
        self.recording = False
        self.discard = False
        self.save = False
        self.block = False
        self.term = False

        #remote commands
        self.block_command=rospy.get_param('/commands/block')
        self.record_command=rospy.get_param('/commands/record')
        self.discard_command=rospy.get_param('/commands/discard')
        self.save_command=rospy.get_param('/commands/save')

        #low_level_commands vars
        self.names_right = rospy.get_param('/arm_right_joints') 
        self.names_left = rospy.get_param('/arm_left_joints')
        self.k_v = rospy.get_param('/gains/k_v')
        self.k_p = rospy.get_param('/gains/k_p')

        #init variables
        ee_init_pos=rospy.get_param('/ee_init_pos')
        ee_init_or=rospy.get_param('/ee_init_or')
        arm_right_init_joint=rospy.get_param('/arm_right_init_joint')
        self.init_joint_pos = np.array(arm_right_init_joint)
        self.curr_pos = np.array(ee_init_pos)
        self.curr_quat = np.array(ee_init_or)
        self.curr_joints_pos = np.array([])
        self.curr_joints_vel = np.array([])
        self.curr_action = np.array([0.0]*3)
        self.grapes_pos = np.array([])
        self.vr_pos = np.array([0.0]*3)
        self.vr_rot = np.array([0.0]*3)
        self.vr_pos_vel = np.array([0.0]*3)
        self.vr_rot_vel = np.array([0.0]*3)
        self.d_vr_pos = np.array([0.0]*3)
        self.target_pos = np.array([0.0]*3)
        self.target_rot = np.array([0.0]*4)

        # init states
        self.init_pos = Point(x=self.curr_pos[0], y=self.curr_pos[1], z=self.curr_pos[2])
        self.init_or = Quaternion(x=self.curr_quat[0], y=self.curr_quat[1], z=self.curr_quat[2], w=self.curr_quat[3])
        
        # ROS stuff
        rospy.init_node("high_level_controller", anonymous=True)

        self.listener = tf.TransformListener()
        rospy.sleep(2)

        rospy.Subscriber('/q2r_right_hand_pose', PoseStamped, self.callback_vr_position_right_arm,queue_size=1) # avg rate > 70 Hz
        #rospy.Subscriber('/q2r_right_hand_twist', Twist, self.callback_vr_orientation_right_arm, queue_size=1)
        rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, self.callback_vr_inputs_right_arm,queue_size=1)
        rospy.Subscriber('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, self.callback_joint_state) #abg rate 1Hz
        rospy.Subscriber('/task_2_value', PoseStamped, self.callback_end_effector_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray,self.callback_grapes_info) #avg rate < 1Hz
        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right)
        
        self.publisher_controller_right = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        self.publisher_rigth_arm = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        rate=rospy.get_param('rates/recording')
        self.control_loop_rate = rospy.Rate(rate) 

        rospy.sleep(2)

        # For saving trajectories
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('imitation_learning')
        task = rospy.get_param('task')
        save_dir = os.path.join(package_path,'data',task)
        self.traj_data = TrajectoryHandler(save_dir,task)

        rospy.loginfo('VRCommands initialized')


    def main(self):
        rospy.loginfo('Beginning...')

        #collecting_init_values
        joints_msg = rospy.wait_for_message('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, timeout=None)
        end_effector_msg = rospy.wait_for_message('/task_2_value', PoseStamped, timeout=None)

        #get the initial pose
        self.curr_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_quat = np.expand_dims(np.array(joints_msg.actual.velocities), 0)
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)

        rospy.loginfo("starting ...")

        # Initialize arm in a standard position
        ee_pos_msg = ExternalReference()
        ee_pos_msg.position = self.init_pos 
        ee_pos_msg.orientation = self.init_or
        self.publisher_controller_right.publish(ee_pos_msg)

        #init vel commands
        velocity_msg_right = Float64MultiArray()
        velocity_msg_left = Float64MultiArray()
        velocity_msg_right.data = [0.0]*7
        velocity_msg_left.data = [0.0]*7

        rospy.sleep(2)

        rospy.loginfo('Ready...')
        t = time.time()
        input('ROS workspace is ready. Press something to start!')

        cnt = 0

        while not rospy.is_shutdown():
            if self.resetting:
                ee_pos_msg.position = Point(x=self.target_pos[0], y=self.target_pos[1], z=self.target_pos[2])
                ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.publisher_controller_right.publish(ee_pos_msg)
                
                joint_command, joint_error = self.joint_PD(self.init_joint_pos)
                velocity_msg_right.data =  joint_command
                self.publisher_rigth_arm.publish(velocity_msg_right)            
                if joint_error < 0.3:
                    print(f'Resetting accomplished: {joint_error}')
                    self.resetting = False
                    self.target_pos = np.array([0.0]*3)
                    self.d_vr_pos = np.array([0.0]*3)
                self.control_loop_rate.sleep()
                continue
            

            if self.discard:
                if self.traj_data.size()>10:
                    print('Trajectory discarded')
                self.traj_data.reset()

            
            elif self.save:
                #print('save')
                self.traj_data.data_external_objs = self.grapes_pos
                trj_n = str(round(datetime.datetime.now().timestamp()))
                self.traj_data.save_trajectory(trj_n)
                self.traj_data.reset()
                self.save = False
            
            elif self.block:
                #print('Blocking')
                velocity_msg_right.data = [0.0]*7
                self.publisher_rigth_arm.publish(velocity_msg_right)

            else:
                action = self.d_vr_pos
                self.target_pos += action

                t = time.time()
                ee_pos_msg.position = Point(x=self.target_pos[0], y=self.target_pos[1], z=self.target_pos[2])
                ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                #ee_pos_msg.orientation = Quaternion(x=self.target_rot[0], y=self.target_rot[1], z=self.target_rot[2], w=self.target_rot[3])
                self.publisher_controller_right.publish(ee_pos_msg)
                    
                #store variables to save
                rec_joints_pos = copy.deepcopy(self.curr_joints_pos)
                rec_joints_vel = copy.deepcopy(self.curr_joints_vel)
                rec_pos_,rec_or_ = self.get_transform(target_frame='base_footprint',source_frame=f'inner_finger_1_right_link')
                
                #wait for the velocity command 
                joint_vel_ik_right = rospy.wait_for_message('/arm_right_forward_velocity_controller/command',Float64MultiArray, timeout=None)
                for i, name in enumerate(self.names_right):
                    index_in_msg = self.names_right.index(name)
                    velocity_msg_right.data[index_in_msg] = joint_vel_ik_right.data[i] * self.k_v
                
                self.publisher_rigth_arm.publish(velocity_msg_right)
                
                #rec variables
                if self.recording>0.0:
                    print("data size: " + str(self.traj_data.size()))
                    rec_joint_action = copy.deepcopy(np.expand_dims(np.array(velocity_msg_right.data), 0))
                    rec_pos = np.expand_dims(np.array(rec_pos_), 0)
                    rec_or = np.expand_dims(np.array(rec_or_), 0)
                    self.traj_data.store_joints_pos(rec_joints_pos)
                    self.traj_data.store_joints_vel(rec_joints_vel)
                    self.traj_data.store_pos(rec_pos)
                    self.traj_data.store_orientation(rec_or)
                    self.traj_data.store_velocity(rec_joint_action)
                    self.traj_data.store_action(np.expand_dims(self.d_vr_pos,0))
                self.control_loop_rate.sleep()

    ##  ------- METHODS ---------


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
            control_velocity = self.k_v * e_i
            error += e_i**2
            joint_vel_command[index_in_msg] = control_velocity
        error = np.sqrt(error)
        return joint_vel_command, error

    def simulator_remove_grape_bunch(self,id_: int):
        rospy.wait_for_service('/simulator')
        cmd = rospy.ServiceProxy('/simulator', Simulator)
        cmd("RemoveGrapeBunch", id_, False, "")



    def get_transform(self, target_frame, source_frame):
        try:
            # Wait for the transform to become available and get the transform
            self.listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return None, None



    ##  -----  CALLBACKS  ------------------------------------------------

    def callback_end_effector_state(self,end_effector_msg):
        pos = end_effector_msg.pose.position
        orient = end_effector_msg.pose.orientation
        self.curr_pos = np.expand_dims(np.array([pos.x,pos.y,pos.z] ), 0)
        self.curr_quat = np.expand_dims(np.array([orient.x,orient.y,orient.z, orient.w]), 0)

    def callback_vr_position_right_arm(self,vr_pose):
        vr_old_pos = copy.deepcopy(self.vr_pos)
        self.vr_pos[0],self.vr_pos[1],self.vr_pos[2] = vr_pose.pose.position.x, -vr_pose.pose.position.y, vr_pose.pose.position.z
        self.d_vr_pos = (self.vr_pos - vr_old_pos)*self.k_p
        self.target_rot[0] = vr_pose.pose.orientation.x
        self.target_rot[1] = vr_pose.pose.orientation.y
        self.target_rot[2] = vr_pose.pose.orientation.z
        self.target_rot[3] = vr_pose.pose.orientation.w

    def callback_vr_orientation_right_arm(self, vr_twist):
        vr_old_rot = copy.deepcopy(np.expand_dims(self.curr_quat[0], -1))
        wx, wy, wz = vr_twist.angular.x, vr_twist.angular.y, vr_twist.angular.z
        delta_theta = np.array([wx, wy, wz])
        delta_quaternion = tf.transformations.quaternion_from_euler(*delta_theta)
        self.target_rot = tf.transformations.quaternion_multiply(vr_old_rot, np.expand_dims(delta_quaternion, -1))
        self.target_rot = self.target_rot / np.linalg.norm(self.target_rot)


    def callback_grapes_info(self, grapes_data):
        ee_pos_1,_ = self.get_transform(target_frame='base_footprint',source_frame=f'fingertip_1_right_link')
        ee_pos_2,_ = self.get_transform(target_frame='base_footprint',source_frame=f'fingertip_2_right_link')
        ee_pos = (np.array(ee_pos_1) + np.array(ee_pos_2))/2 
        poses = []
        for box in grapes_data.boxes:
            i = box.index
            g_pos, _ = self.get_transform(target_frame='base_footprint',source_frame=f'Bunch_{i}')
            poses.append(g_pos)
            dist = np.linalg.norm(np.array(ee_pos)-np.array(g_pos))
            if dist < 0.1:
                self.simulator_remove_grape_bunch(int(i))
                print(f'\n{i} grape removed')
        self.grapes_pos = np.expand_dims(np.array(poses),0)
    
    
    def callback_vr_inputs_right_arm(self,vr_inputs):

        curr_commands = {
            'A': vr_inputs.button_lower,
            'B': vr_inputs.button_upper,
            'trigger': vr_inputs.press_index,
            'middle': vr_inputs.press_middle,
        }

        self.discard = curr_commands[self.discard_command] 
        self.recording = curr_commands[self.record_command]  
        self.block = curr_commands[self.block_command] 
        self.save = curr_commands[self.save_command] 

        if curr_commands['A'] and curr_commands['B']:
            self.resetting = True

    def callback_joint_state(self,joints_msg):
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)

if __name__ == "__main__":
    try:
        node = VRCommands()
        node.main()
    except rospy.ROSInterruptException:
        pass


