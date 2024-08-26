#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
import random
from geometry_msgs.msg import Pose
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
from tf.transformations import euler_from_quaternion, quaternion_inverse, quaternion_multiply, quaternion_from_euler

# import the utils libraries
rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path, '/src/utils')
sys.path.append(module_path)
from utils import Buffer, simulator_remove_grape_bunch, communicate_instructions, get_closest_obj, check_workspace
from geometry_msgs.msg import Twist


class VRCommands:
    def __init__(self):
       
        # status variables
        self.recording, self.discard, self.save, self.block, self.resetting = False, False, False, False, False

        self.clip_rotations = True
        self.random_start = True
        self.torso_height = rospy.get_param('torso_height')

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
        self.k_r = rospy.get_param('/gains/k_r')
        rospy.sleep(2)

        # init variables
        self.init_pose = np.array(rospy.get_param('init_pose'))
        self.vr_pos, self.ee_pos, self.ee_rot, self.d_vr_pos = np.array([0.]*3),np.array([0.]*3),np.array([0.]*3),np.array([0.]*3)
        self.threshold = rospy.get_param('/threshold')
        self.d_vr_rot = np.array([0.]*3)
        self.vr_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.target_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.ee_pos_msg = ExternalReference()
        self.grapes_pos, self.grapes_idx, self.removed_grape = [],[],[]
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0] * 8


        # ROS stuff
        rospy.init_node("high_level_controller", anonymous=True)

        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right+['torso_lift_joint'])
        rospy.set_param('canopies_simulator/joint_states/rate', 100)
        self.listener = tf.TransformListener()

        self.publisher_mobile_base = rospy.Publisher('/canopies_simulator/moving_base/twist', Twist, queue_size=10)
        self.publisher_controller_right = rospy.Publisher('/external_references_for_right_arm', ExternalReference,queue_size=1)
        self.publisher_joint_commands = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command',
                                                   Float64MultiArray, queue_size=1)
        self.publisher_teleport = rospy.Publisher('/canopies_simulator/moving_base/teleport',
                                                   Pose, queue_size=1)

        rospy.Subscriber('/q2r_right_hand_pose', PoseStamped, self.callback_vr_position_right_arm, queue_size=1)
        rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, self.callback_vr_inputs_right_arm, queue_size=1)
        rospy.Subscriber('/canopies_simulator/joint_states', JointState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes, queue_size=1)
        rospy.Subscriber('/direct_kinematics/rigth_arm_end_effector', PoseStamped, self.callback_end_effector, queue_size=1)
        rospy.Subscriber('/arm_right_forward_velocity_controller/command', Float64MultiArray, self.callback_velocity_commands, queue_size=1)

        rate = rospy.get_param('rates/recording')
        self.control_loop_rate = rospy.Rate(rate)
        rospy.sleep(2)

        # For saving trajectories
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('imitation_learning')
        self.task = rospy.get_param('task')
        data_folder = rospy.get_param('folders/data')
        save_dir = os.path.join(package_path, data_folder, self.task)
        self.traj_buffer = Buffer(save_dir)
        rospy.loginfo('VRCommands initialized')


    def main(self):

        if self.random_start:
            target_id = self.random_init()
            self.send_viapoint(self.init_pose[:3], self.init_pose[3:])
            rospy.sleep(5)
        else:
            target_id, _ = get_closest_obj(np.array([0.,0.,0.]), self.grapes_pos, self.grapes_idx)
            self.send_viapoint(self.init_pose[:3], self.init_pose[3:])


        #adjust the torso height
        goal_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{target_id}')
        head, _ = self.get_transform(target_frame='base_footprint', source_frame='head_2_link')
        self.head_height_controller(goal_pos[-1] - self.torso_height)
        rospy.sleep(1)

        rospy.loginfo(f'ROS workspace is ready for task {self.task}. Press the lateral button to start!')

        while not self.recording:
            rospy.sleep(0.1)

        ## ====================================== SIM LOOP ========================================================
        target_pos, target_rot = self.init_pose[:3], self.init_pose[3:]
        offset = self.init_pose[:3] - copy.deepcopy(self.vr_pos)
        offset_rot = quaternion_inverse(copy.deepcopy(self.vr_rot))

        self.sim_step=0
        while not rospy.is_shutdown():

            #checks
            target_id, target_dist = self.grape_remove_check(self.ee_pos)

            if self.sim_step % 50 == 0:
                communicate_instructions(target_id, target_dist)

            if self.resetting:
                print('resetting...')
                self.send_viapoint(self.init_pose[:3], self.init_pose[3:])

            elif self.discard:
                rospy.loginfo('Trajectory discarded')
                self.traj_buffer.reset()

            elif self.save:
                if self.traj_buffer.size() > 10:
                    self.traj_buffer.grapes_positions = copy.deepcopy(list(self.grapes_pos))
                    self.traj_buffer.show_current_status()
                    save_path = self.traj_buffer.save_trajectory(f'{str(round(datetime.datetime.now().timestamp()))}.pkl')
                    rospy.loginfo(f'Trajectory saved in {save_path}')
                    self.traj_buffer.reset()
                    self.save = False

            elif self.block:
                self.velocity_msg_right.data = [0.0] * 8
                self.publisher_joint_commands.publish(self.velocity_msg_right)

            else:

                #compute the target position
                new_target_pos = offset + self.vr_pos
                d_target_pos = copy.deepcopy(new_target_pos - target_pos)
                target_pos = new_target_pos

                # compute the target orientation
                target_rot = quaternion_multiply(self.vr_rot, offset_rot)
                if self.clip_rotations:
                    target_euler = np.clip(
                        euler_from_quaternion(target_rot), a_min=-np.pi/3, a_max=np.pi/3
                    )*self.k_r
                    target_rot = quaternion_from_euler(*target_euler)
                if self.vr_mirroring:
                    target_rot *= [1., 1., -1., -1.]

                # send the commanded pose
                self.send_viapoint(target_pos, target_rot)
                t = rospy.get_time()

                if self.recording:
                    self.traj_buffer.target_positions['value'].append(list(target_pos))
                    self.traj_buffer.target_orientations['value'].append(list(target_rot))
                    self.traj_buffer.command_positions['value'].append(list(d_target_pos))
                    self.traj_buffer.target_positions['time'].append(t)
                    self.traj_buffer.target_orientations['time'].append(t)
                    self.traj_buffer.command_positions['time'].append(t)

            self.sim_step += 1
            self.control_loop_rate.sleep()



    ## ====================================== CALLBACKS ==========================================================

    def callback_vr_position_right_arm(self, vr_pose):
        t = rospy.get_time()
        mirror = -1. if self.vr_mirroring else 1.

        self.vr_pos = np.array(
            [mirror*vr_pose.pose.position.x,
            mirror*vr_pose.pose.position.y,
            vr_pose.pose.position.z]
        )
        self.vr_rot = np.array(
            [vr_pose.pose.orientation.x,
            vr_pose.pose.orientation.y,
            vr_pose.pose.orientation.z,
            vr_pose.pose.orientation.w]
        )

    def callback_velocity_commands(self, msg_vel):
        if self.block:
            self.velocity_msg_right.data = [0.0] * 8
        else:
            for i, name in enumerate(self.names_right):
                index_in_msg = self.names_right.index(name)
                self.velocity_msg_right.data[index_in_msg] = msg_vel.data[i] * self.k_v
        self.publisher_joint_commands.publish(self.velocity_msg_right)

    def callback_grapes(self, grapes_data):
        grapes_pos, grapes_idx = [], []
        for box in grapes_data.boxes:
            grapes_idx.append(box.index)
            g_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{box.index}')
            g_pos = np.add(g_pos, [0.,0.,-0.1])
            grapes_pos.append(g_pos)
        self.grapes_pos = np.array(grapes_pos)
        self.grapes_idx = grapes_idx

    def callback_vr_inputs_right_arm(self, vr_inputs):

        curr_commands = {
            'A': vr_inputs.button_lower,
            'B': vr_inputs.button_upper,
            'trigger': vr_inputs.press_index,
            'middle': vr_inputs.press_middle,
        }
        self.resetting = vr_inputs.button_upper and vr_inputs.press_middle
        if self.resetting:
            self.discard, self.recording, self.block, self.save = [False]*4
        else:
            self.discard = curr_commands[self.discard_command]
            self.recording = curr_commands[self.record_command]
            self.block = curr_commands[self.block_command]
            self.save = curr_commands[self.save_command]

        self.mobile_base_controller(vr_inputs.thumb_stick_vertical, -vr_inputs.thumb_stick_horizontal)

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


    def callback_end_effector(self, pose_msg):
        t = pose_msg.header.stamp.secs + pose_msg.header.stamp.nsecs/(10**9)
        self.ee_pos = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        self.ee_rot = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]
        if self.recording:
            self.traj_buffer.cartesian_positions['value'].append(self.ee_pos)
            self.traj_buffer.cartesian_orientations['value'].append(self.ee_rot)
            self.traj_buffer.cartesian_positions['time'].append(t)
            self.traj_buffer.cartesian_orientations['time'].append(t)


    ## ====================================== OTHER METHODS ==========================================================
    def grape_remove_check(self, ee_pos):
        target_id, target_dist = get_closest_obj(ee_pos, self.grapes_pos, self.grapes_idx)
        dists = np.linalg.norm(self.grapes_pos - np.array(ee_pos), axis=-1)
        threshold = (dists < self.threshold)*1.
        for i in np.nonzero(threshold)[0]:
            #if self.grapes_idx[i] not in self.removed_grapes:
            rospy.loginfo(f'\n{self.grapes_idx[i]} grape removed')
                #simulator_remove_grape_bunch(int(self.grapes_idx[i]))
                #self.removed_grapes.append(self.grapes_idx[i])

        return target_id, target_dist

    def mobile_base_controller(self, v, u):
        twist = Twist()
        mirror = -1 if self.vr_mirroring else 1
        twist.linear.x = v * mirror
        twist.angular.z = u * mirror
        self.publisher_mobile_base.publish(twist)

        '''
        def torso_controller(self, h):
        """
        Moves the torso joint until a specif height (considering head_2_link wrt base_frame as reference)
        """
        x_0, _ = self.get_transform(target_frame='base_footprint', source_frame='head_2_link')
        e, z = 1., 0.
        while abs(e)>0.001:
            self.velocity_msg_right.data = [0.0] * 7 + [0.1 * np.sign(e)]
            rospy.sleep(0.01)
            x, _ = self.get_transform(target_frame='base_footprint', source_frame='head_2_link')
            z = x[2] - x_0[2]
            e = h-z
        self.velocity_msg_right.data = [0.0] * 8
        self.publisher_joint_commands.publish(self.velocity_msg_right)
        return z'''

    def random_init(self):
        """
        Teleport the robot to a randomized location such that there is a reachable grape
        """
        workspace = rospy.get_param('workspace_cylinder')
        _, _, z_t = rospy.get_param('mobile_base_init_pos')
        grape_pos = np.array(rospy.get_param('grape_27_position'))
        mb_offset = np.array(rospy.get_param('mobile_base_offset'))

        r = random.uniform(*workspace['radius_bounds'])
        theta = random.uniform(*(np.array(workspace['theta_bounds'])+np.pi))
        noise = (r*np.cos(theta), r*np.sin(theta), 0.)
        target_pos = grape_pos + mb_offset - workspace['centre'] + noise
        target_pos[2] = z_t

        pose = Pose()
        pose.position = Point(*target_pos)
        pose.orientation = Quaternion(*quaternion_from_euler(0., 0., 0.))
        self.publisher_teleport.publish(pose)
        rospy.loginfo(f'Teleported to {target_pos}')
        return 27

    def get_transform(self, target_frame, source_frame):
        try:
            # Wait for the transform to become available and get the transform
            self.listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return None, None

    def head_height_controller(self, h):
        """
        Reach the head height wrt base_footprint lifting the torso lift joint
        """
        e, z = 1., 0.
        while abs(e)>0.001:
            self.velocity_msg_right.data = [0.0] * 7 + [0.1 * np.sign(e)]
            rospy.sleep(0.01)
            (_, _, z), _ = self.get_transform(target_frame='base_footprint', source_frame='head_2_link')
            e = h - z

        self.velocity_msg_right.data = [0.0] * 8
        self.publisher_joint_commands.publish(self.velocity_msg_right)
        return z

    def send_viapoint(self, position, orientation):
        """
        INPUT: viapoint postion (3D) and orientation (quaternion)
        """
        self.ee_pos_msg.position = Point(x=position[0], y=position[1], z=position[2])
        self.ee_pos_msg.orientation = Quaternion(x=orientation[0],y=orientation[1],z=orientation[2],w=orientation[3])
        self.publisher_controller_right.publish(self.ee_pos_msg)


if __name__ == "__main__":
    try:
        node = VRCommands()
        node.main()
    except rospy.ROSInterruptException:
        pass


