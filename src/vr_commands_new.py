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
# import the utils libraries
import rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path, '/src/utils')
sys.path.append(module_path)
from utils import TrajectoryHandler


class VRCommands:
    def __init__(self):

       
        # status variables
        self.resetting, self.recording, self.discard, self.save, self.block = False, False, False, False, False

        # remote commands
        self.block_command = rospy.get_param('/vr_commands/block')
        self.record_command = rospy.get_param('/vr_commands/record')
        self.discard_command = rospy.get_param('/vr_commands/discard')
        self.save_command = rospy.get_param('/vr_commands/save')
        self.vr_mirroring = rospy.get_param('/vr_commands/mirroring')

        # low_level_commands vars
        self.names_right = rospy.get_param('/arm_right_joints')
        self.names_left = rospy.get_param('/arm_left_joints')
        self.k_v = rospy.get_param('/gains/k_v')
        self.k_p = rospy.get_param('/gains/k_p')
        rospy.sleep(2)

        # init variables
        self.init_pos = np.array(rospy.get_param('ee_init_pos'))
        self.init_or = np.array(rospy.get_param('ee_init_or'))
        self.init_joint_pos = rospy.get_param('arm_right_init_joint')
        self.vr_pos = np.array([0.0] * 3)
        self.d_vr_pos = np.array([0.0] * 3)
        self.vr_pos_stack = []
        self.vr_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.target_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.curr_joints_pos, self.curr_joints_vel = np.zeros((1, 7)), np.zeros((1, 7))
        self.threshold = rospy.get_param('/threshold')
        self.grapes_pos, self.grapes_idx, self.removed_grapes = [],[],[]

        # ROS stuff
        rospy.init_node("high_level_controller", anonymous=True)
        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right)
        rospy.set_param('canopies_simulator/joint_states/rate', 100)

       
        self.listener = tf.TransformListener()
        rospy.Subscriber('/q2r_right_hand_pose', PoseStamped, self.callback_vr_position_right_arm, queue_size=1)
        rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, self.callback_vr_inputs_right_arm, queue_size=1)
        #rospy.Subscriber('/q2r_right_hand_twist', Twist, self.callback_vr_orientation_right_arm, queue_size=1)
        rospy.Subscriber('/canopies_simulator/joint_states', JointState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes, queue_size=1)
        self.publisher_controller_right = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        self.publisher_rigth_arm = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        self.publisher_torso = rospy.Publisher('/canopies_simulator/torso_controller/command', JointTrajectory, queue_size=1)
        rate = rospy.get_param('rates/recording')
        self.control_loop_rate = rospy.Rate(rate)
        rospy.sleep(2)


        # For saving trajectories
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('imitation_learning')
        self.task = rospy.get_param('task')
        data_folder = rospy.get_param('folders/data')
        save_dir = os.path.join(package_path, data_folder, self.task)
        self.traj_data = TrajectoryHandler(save_dir)
        rospy.loginfo('VRCommands initialized')



        print('wating for q2r msg ....')
        vr_msg = rospy.wait_for_message('/q2r_right_hand_pose', Pose, timeout=None)
        self.vr_pos = np.array([vr_msg.pose.position.x, vr_msg.pose.position.y, vr_msg.pose.position.z])
        self.vr_pos_stack.clear()


        #TORSO
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.joint_names = ['torso_base_joint']

        # Create a JointTrajectoryPoint
        point = JointTrajectoryPoint()
        point.positions = [1.0, 2.0, 3.0]
        point.velocities = [0.5, 0.5, 0.5]
        point.accelerations = [0.1, 0.1, 0.1]
        point.effort = [0.0, 0.0, 0.0]
        point.time_from_start = rospy.Duration(1.0)

        # Add the point to the trajectory
        joint_traj.points.append(point)


    def main(self):


        ## ----------------- SETUP -----------------

        # collecting_init_values
        #joints_msg = rospy.wait_for_message('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, timeout=None)
        #self.curr_joints_pos = np.expand_dims(np.array(joints_msg.actual.positions), 0)
        #self.curr_joints_vel = np.expand_dims(np.array(joints_msg.actual.velocities), 0)

        # init vel commands
        velocity_msg_right = Float64MultiArray()
        velocity_msg_left = Float64MultiArray()
        velocity_msg_right.data = [0.0] * 7
        velocity_msg_left.data = [0.0] * 7

        # Initialize arm in a standard position
        ee_pos_msg = ExternalReference()
        ee_pos_msg.position = Point(x=self.init_pos[0], y=self.init_pos[1], z=self.init_pos[2]) #self.init_pos
        ee_pos_msg.orientation = Quaternion(x=self.init_or[0], y=self.init_or[1], z=self.init_or[2], w=self.init_or[3]) #self.init_or
        self.publisher_controller_right.publish(ee_pos_msg)
        target_pos, target_or = self.init_pos, self.init_or
        rec_pos, rec_or = self.get_transform(target_frame='base_footprint', source_frame=f'inner_finger_1_right_link')
        

        rospy.sleep(1)
        
        rospy.loginfo(f'Beginning task "{self.task}" ...')

        ## ---------------- SIM LOOP ----------------
        #input('ROS workspace is ready. Press something to start!')
        self.sim_step=0
        while not rospy.is_shutdown():

            if self.discard:
                if self.traj_data.size() > 10:
                    rospy.loginfo('Trajectory discarded')
                self.traj_data.reset()

            elif self.save:
                trj_n = str(round(datetime.datetime.now().timestamp()))
                save_path = self.traj_data.save_trajectory(trj_n)
                rospy.loginfo(f'Trajectory saved in {save_path}')
                self.traj_data.reset()
                self.save = False

            elif self.block:
                velocity_msg_right.data = [0.0] * 7
                self.publisher_rigth_arm.publish(velocity_msg_right)

            else:
                act = self.compute_vr_command()
                target_pos += np.array([0.]*3)

                ee_pos_msg.position = Point(x=target_pos[0], y=target_pos[1], z=target_pos[2])
                ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
                #ee_pos_msg.orientation = Quaternion(x=self.target_rot[0], y=self.target_rot[1], z=self.target_rot[2], w=self.target_rot[3])
                self.publisher_controller_right.publish(ee_pos_msg)

                # store variables to save
                rec_joints_pos = copy.deepcopy(self.curr_joints_pos)
                rec_joints_vel = copy.deepcopy(self.curr_joints_vel)
                rec_pos, rec_or = self.get_transform(target_frame='base_footprint', source_frame=f'inner_finger_1_right_link')
                self.grape_revove_check(rec_pos)

                # wait for the velocity command
                joint_vel_ik_right = rospy.wait_for_message('/arm_right_forward_velocity_controller/command', Float64MultiArray, timeout=None)
                for i, name in enumerate(self.names_right):
                    index_in_msg = self.names_right.index(name)
                    velocity_msg_right.data[index_in_msg] = joint_vel_ik_right.data[i] * self.k_v

                self.publisher_rigth_arm.publish(velocity_msg_right)
                # rec variables
                if self.recording > 0.0:
                    self.show_update(f"data size: {self.traj_data.size()}")

                    self.traj_data.store_data(
                        (
                            rec_joints_pos,
                            rec_joints_vel,
                            np.expand_dims(rec_pos, 0),
                            np.expand_dims(rec_or, 0),
                            copy.deepcopy(np.expand_dims(np.array(velocity_msg_right.data), 0)),
                            np.expand_dims(self.d_vr_pos, 0),
                            self.grapes_pos
                        )
                    )
            self.sim_step+=1
            self.control_loop_rate.sleep()


    def simulator_remove_grape_bunch(self, id_: int):
        rospy.wait_for_service('/simulator')
        cmd = rospy.ServiceProxy('/simulator', Simulator)
        cmd("RemoveGrapeBunch", id_, False, "")

    def get_transform(self, target_frame, source_frame):
        try:
            # Wait for the transform to become available and get the transform
            self.listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return np.array(trans), np.array(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return None, None

    ##  -----  CALLBACKS  ------------------------------------------------

    def callback_vr_position_right_arm(self, vr_pose):
        mirror = -1. if self.vr_mirroring else 1.
        self.vr_pos_stack.append(np.array([vr_pose.pose.position.x, mirror*vr_pose.pose.position.y, vr_pose.pose.position.z]))
        '''vr_old_pos = copy.deepcopy(self.vr_pos)
        mirror = -1. if self.vr_mirroring else 1.
        self.vr_pos[0], self.vr_pos[1], self.vr_pos[2] = vr_pose.pose.position.x, mirror*vr_pose.pose.position.y, vr_pose.pose.position.z
        self.d_vr_pos = (self.vr_pos - vr_old_pos) * self.k_p'''


    def callback_grapes(self, grapes_data):
        grapes_pos, grapes_idx = [], []
        for box in grapes_data.boxes:
            grapes_idx.append(box.index)
            g_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{box.index}')
            grapes_pos.append(g_pos)
        self.grapes_pos = np.array(grapes_pos)
        self.grapes_idx = grapes_idx

    def callback_vr_orientation_right_arm(self, vr_twist):
        vr_old_rot = copy.deepcopy(np.expand_dims(self.vr_rot, -1))
        wx, wy, wz = vr_twist.angular.x, vr_twist.angular.y, vr_twist.angular.z
        delta_theta = np.array([wx, wy, wz])
        delta_quaternion = tf.transformations.quaternion_from_euler(*delta_theta)
        self.target_rot = tf.transformations.quaternion_multiply(vr_old_rot, np.expand_dims(delta_quaternion, -1))
        #rospy.loginfo(f"target_rot before: {self.target_rot}")
        self.target_rot = self.target_rot / np.linalg.norm(self.target_rot)
        self.vr_rot = self.target_rot[:, 0]
        #rospy.loginfo(f"target_rot: {self.vr_rot}")

    def callback_vr_inputs_right_arm(self, vr_inputs):

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

    def callback_joint_state(self, joints_msg):

        for name in self.names_right:
            index_in_joint_state = joints_msg.name.index(name)
            index_in_msg = self.names_right.index(name)
            self.curr_joints_pos[0, index_in_msg] = joints_msg.position[index_in_joint_state]
            self.curr_joints_vel[0, index_in_msg] = joints_msg.velocity[index_in_joint_state]

    ## OTHER METHODS
    def grape_revove_check(self, ee_pos):
        dists = np.linalg.norm(np.array(ee_pos) - self.grapes_pos, axis=-1)
        if self.sim_step%50==0:
            self.show_update(f'{self.grapes_idx[np.argmin(dists)]}: {np.min(dists)}')
        threshold = (dists < self.threshold)*1.
        for i in np.nonzero(threshold)[0]:
            if self.grapes_idx[i] not in self.removed_grapes:
                rospy.loginfo(f'\n{self.grapes_idx[i]} grape removed')
                #self.simulator_remove_grape_bunch(int(self.grapes_idx[i]))
                #self.removed_grapes.append(self.grapes_idx[i])

    def register_grapes(self):
        grapes_pos, grapes_idx = [], []
        grapes_data = rospy.wait_for_message('/canopies_simulator/grape_boxes', BoundBoxArray, timeout=None)
        for box in grapes_data.boxes:
            grapes_idx.append(box.index)
            g_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{box.index}')
            grapes_pos.append(g_pos)
        grapes_pos = np.array(grapes_pos)
        return grapes_pos, grapes_idx, []

    def show_update(self, a, n=50):
        if self.sim_step%n==0:
            rospy.loginfo(a)

    def compute_vr_command(self):
        old_vr_pos = copy.deepcopy(self.vr_pos)
        curr_pos = np.mean(np.stack(self.vr_pos_stack), axis=0)
        dpos = (old_vr_pos - curr_pos)*self.k_p
        self.vr_pos = copy.deepcopy(curr_pos)
        self.vr_pos_stack.clear()
        return dpos



if __name__ == "__main__":
    try:
        node = VRCommands()
        node.main()
    except rospy.ROSInterruptException:
        pass






'''
/* -------------------------------- Orientation -------------------------------- 

    Re = quat2rot(Eigen::Vector4d(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w));
    (this->task_2_desired_value.tail(4) = this->task_2_initial_value.tail(4) + rot2quat(Re*(Re.transpose()*Re_des)));*/

'''