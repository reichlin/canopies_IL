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


        

class VRCommands:
    def __init__(self): 
        
        #status variables
        self.resetting = False
        self.recording = False
        self.discard = False
        self.save = False
        self.block = False
        self.term = False

        #low_level_commands vars
        self.names_right = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
        self.names_left = ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint"]
        self.k_v = 20
        self.k_p = 4

        # init states
        self.init_pos = Point(x=0.093, y=0.1, z=-0.0728)
        self.init_or = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.init_joint_pos = np.array([-0.780468,  1.467391,  1.142556,  1.570801, -0.121956,  1.398000, 0.0])

        #init variables
        self.curr_pos = np.array([0.093,0.1,-0.0728])
        self.curr_or = np.array([])
        self.curr_joints_pos = np.array([])
        self.curr_joints_vel = np.array([])
        self.curr_action = np.array([0,0,0])
        self.grapes_pos = np.array([])
        self.vr_pos = np.array([0.0, 0.0, 0.0])
        self.vr_rot = np.array([0.0, 0.0, 0.0])
        self.vr_pos_vel = np.array([0.0, 0.0, 0.0])
        self.vr_rot_vel = np.array([0.0, 0.0, 0.0])
        self.d_vr_pos = np.array([0.0, 0.0, 0.0])

        # ROS stuff
        rospy.init_node("vr_commands", anonymous=True)

        self.listener = tf.TransformListener()
        rospy.sleep(2)


        rospy.Subscriber('/q2r_right_hand_pose', PoseStamped, self.callback_vr_position_right_arm,queue_size=1) # avg rate > 70 Hz
        rospy.Subscriber('/q2r_right_hand_twist', Twist, self.callback_vr_velocity_right_arm,queue_size=1)
        rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, self.callback_vr_inputs_right_arm,queue_size=1)
        rospy.Subscriber('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, self.callback_joint_state) #abg rate 1Hz
        rospy.Subscriber('/task_2_value', PoseStamped, self.callback_end_effector_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray,self.callback_grapes_info) #avg rate < 1Hz
        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right)
        
        self.publisher_controller_right = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        self.publisher_rigth_arm = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        
        self.control_loop_rate = rospy.Rate(50)  # 10Hz

        rospy.sleep(2)

        # For saving trajectories
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('imitation_learning')
        save_dir = os.path.join(package_path,'data')
        self.traj_data = TrajectoryHandler(save_dir)

        rospy.loginfo('VRCommands initialized')


    def main(self):
        rospy.loginfo('Beginning...')

        #collecting_init_values
        joints_msg = rospy.wait_for_message('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, timeout=None)
        end_effector_msg = rospy.wait_for_message('/task_2_value', PoseStamped, timeout=None)

        #get the initial pose
        self.curr_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_or = np.expand_dims(np.array(joints_msg.actual.velocities), 0)
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)

        rospy.loginfo("starting ...")

        # Initialize arm in a standard position
        ee_pos_msg = ExternalReference()
        ee_pos_msg.position = self.init_pos 
        ee_pos_msg.orientation = self.init_or
        self.publisher_controller_right.publish(ee_pos_msg)
        self.target_pos = np.array([0.093, 0.1, -0.0728])

        #init vel commands
        velocity_msg_right = Float64MultiArray()
        velocity_msg_left = Float64MultiArray()
        velocity_msg_right.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        velocity_msg_left.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        rospy.sleep(2)

        # get init frame TF
        #pos_base_init, quad_base_init = self.get_transform(target_frame='inner_finger_1_right_link',source_frame=f'base_footprint')        #pos_base_init, quad_base_init = self.get_transform(target_frame='inner_finger_1_right_link',source_frame=f'base_footprint')
        pos_base_init, quad_base_init = self.get_transform(target_frame='base_footprint',source_frame=f'inner_finger_1_right_link')        #pos_base_init, quad_base_init = self.get_transform(target_frame='inner_finger_1_right_link',source_frame=f'base_footprint')
        pos_base_init = [0.093, 0.1, -0.0728]
        self.tf_base_init = construct_transformation_matrix(pos_base_init, [0,0,0,1]) 

        rospy.loginfo('Ready...')
        t = time.time()
        grape_pos = find_closest_position([0,0,0],self.grapes_pos)

        #offset, _ = self.get_transform(target_frame='inner_finger_1_right_link',source_frame=f'arm_right_7_link')        #pos_base_init, quad_base_init = self.get_transform(target_frame='inner_finger_1_right_link',source_frame=f'base_footprint')
        self.target_pos = np.array([0.0]*3)#np.array([0.093, 0.1, -0.0728]) 
        ee_pos_init, _ = self.get_transform(target_frame='base_footprint',source_frame=f'arm_right_7_link')       

        #self.resetting = True

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
                self.traj_data.store_external_objs(self.grapes_pos)
                trj_n = str(round(datetime.datetime.now().timestamp()))
                self.traj_data.save_trajectory(trj_n)
                self.traj_data.reset()
                self.save = False
            
            elif self.block:
                #print('Blocking')
                velocity_msg_right.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.publisher_rigth_arm.publish(velocity_msg_right)

            else:
                action = self.d_vr_pos
                self.target_pos += action

                t = time.time()
                ee_pos_msg.position = Point(x=self.target_pos[0], y=self.target_pos[1], z=self.target_pos[2])
                ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
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
                    rec_joint_action = copy.deepcopy(np.expand_dims(np.array(velocity_msg_right.data), 0))
                    rec_pos = np.expand_dims(np.array(rec_pos_), 0)
                    rec_or = np.expand_dims(np.array(rec_or_), 0)
                    self.traj_data.store_joints_pos(rec_joints_pos)
                    self.traj_data.store_joints_vel(rec_joints_vel)
                    self.traj_data.store_pos(rec_pos)
                    self.traj_data.store_orientation(rec_or)
                    self.traj_data.store_velocity(rec_joint_action)
                    self.traj_data.store_vr_action(np.expand_dims(self.d_vr_pos,0))
                self.control_loop_rate.sleep()

    ##  ------- METHODS ---------

    def project_pos(self, pos):
        pos_homo = np.array([*pos,1])
        #tf_pos = np.eye(4)
        #tf_pos[3,:3] = pos
        #pos_proj = np.dot(tf_pos, self.tf_base_init)
        #return pos_proj[3,:3]
        return (self.tf_base_init @ pos_homo)[:3]

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
        self.curr_or = np.expand_dims(np.array([orient.x,orient.y,orient.z, orient.w]), 0)

    def callback_vr_velocity_right_arm(self,vr_twist):
        pass 
        #vr_pos_vel[0], vr_pos_vel[1], vr_pos_vel[2] = vr_twist.linear.x, vr_twist.linear.y, vr_twist.linear.z
        #vr_rot_vel = vr_twist.angular


    def callback_vr_position_right_arm(self,vr_pose):
        vr_old_pos = copy.deepcopy(self.vr_pos)
        self.vr_pos[0],self.vr_pos[1],self.vr_pos[2] = vr_pose.pose.position.x, vr_pose.pose.position.y, vr_pose.pose.position.z
        self.vr_rot = vr_pose.pose.orientation
        self.d_vr_pos = (self.vr_pos - vr_old_pos)*self.k_p


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
            if dist<0.1:
                #self.simulator_remove_grape_bunch(int(box.index))
                print(f'{i} grape removed')
        self.grapes_pos = poses
    
    
    def callback_vr_inputs_right_arm(self,vr_inputs):
        # A - Save traj
        # B - Discard traj
        # mid - REC
        # index - block
        # A+B - resetting
        
        self.discard = vr_inputs.button_upper
        self.recording = vr_inputs.press_index 
        self.block = vr_inputs.press_middle
        self.save = vr_inputs.button_lower

        if self.discard and self.save:
            self.resetting = True

    def callback_joint_state(self,joints_msg):
        self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)



def construct_transformation_matrix(trans, rot):
    """ Construct the 4x4 transformation matrix from translation and quaternion rotation. """
    r = R.from_quat(rot)
    rot_matrix = r.as_matrix()
    
    # Insert the translation into the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3,:3] = rot_matrix
    transformation_matrix[:3, 3] = trans
    
    return transformation_matrix

def find_closest_position(a, positions):
    min_distance = float('inf')
    closest_position = None
    
    for pos in positions:
        pos = np.array(pos)
        distance = np.linalg.norm(a - pos)
        if distance < min_distance and pos[0]>0 and pos[1]<0:
            min_distance = distance
            closest_position = pos
            
    return closest_position

class TrajectoryHandler:
    def __init__(self, save_dir:str):
        self.init_data_arrays()
        self.tag = 'grasp' #push'
        self.cnt=0
        self.save_dir = save_dir
        # "/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data"
        os.makedirs(self.save_dir,exist_ok=True)

    def init_data_arrays(self):
        self.data_joints_pos = []
        self.data_joints_vel = []
        self.data_pos = []
        self.data_or = []
        self.data_joint_act = []
        self.data_vr_act = []

    def size(self):
        return len(self.data_joints_pos) 

    def store_joints_pos(self, joints_pos):
        #self.data_empty = False
        self.data_joints_pos.append(joints_pos)

    def store_joints_vel(self, joints_vel):
        self.data_joints_vel.append(joints_vel)

    def store_pos(self, pos):
        self.data_pos.append(pos)

    def store_orientation(self, orientation):
        self.data_or.append(orientation)

    def store_velocity(self, vel):
        self.data_joint_act.append(vel)
    
    def store_external_objs(self, objects_pos):
        #self.data_external_objs = []
        #for box in objects:
        #    self.data_external_objs.append(np.array([[[box.xmin, box.xmax], [box.ymin, box.ymax], [box.zmin, box.zmax]]]))
        self.data_external_objs = np.concatenate(objects_pos, 0)
    
    def store_vr_action(self, vel):
        self.data_vr_act.append(vel)

    def show_current_status(self):
        print('---')
        print('data_joints_pos: ', len(self.data_joints_pos))
        print('data_joints_vel: ', len(self.data_joints_vel))
        print('data_pos: ', len(self.data_pos))
        print('data_or: ', len(self.data_or))
        print('data_joint_act: ', len(self.data_joint_act))
        print('data_vr_act: ', len(self.data_vr_act))

    def save_trajectory(self, name):
        if self.size()>100:
            path = os.path.join(self.save_dir,self.tag)
            os.makedirs(path, exist_ok=True)
            file = os.path.join(path,f"traj_{name}.npz")
            np.savez(file,
                np.concatenate(self.data_joints_pos, 0),
                np.concatenate(self.data_joints_vel, 0),
                np.concatenate(self.data_pos, 0),
                np.concatenate(self.data_or, 0),
                np.concatenate(self.data_joint_act, 0),
                self.data_external_objs,
                np.concatenate(self.data_vr_act,0), 
            )
            self.cnt+=1
            print(f'Trajectory ({self.cnt}) of {len(self.data_joints_pos)} steps saved in {file}')
            self.show_current_status()
        
    def reset(self):
        self.init_data_arrays()

if __name__ == "__main__":
    try:
        node = VRCommands()
        node.main()
    except rospy.ROSInterruptException:
        pass


