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
from canopies_simulator.srv import Simulator
# import the utils libraries
rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path, '/src/utils')
sys.path.append(module_path)
from utils import TrajectoryHandler, Buffer
from geometry_msgs.msg import Twist


class VRCommands:
    def __init__(self):

       
        # status variables
        self.recording, self.discard, self.save, self.block, self.resetting = False, False, False, False, False

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
        self.init_rot = np.array(rospy.get_param('ee_init_or'))
        self.init_joint_pos = rospy.get_param('arm_right_init_joint')
        self.vr_pos, self.ee_pos, self.ee_rot, self.d_vr_pos = np.array([0.]*3),np.array([0.]*3),np.array([0.]*3),np.array([0.]*3)
        self.threshold = rospy.get_param('/threshold')
        self.d_vr_rot = np.array([0.]*3)
        self.vr_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.target_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.grapes_pos, self.grapes_idx, self.removed_grape = [],[],[]
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0] * 8


        # ROS stuff
        rospy.init_node("high_level_controller", anonymous=True)
        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right+['torso_lift_joint'])
        rospy.set_param('canopies_simulator/joint_states/rate', 100)
        self.publisher_mobile_base = rospy.Publisher('/canopies_simulator/moving_base/twist', Twist, queue_size=10)
        self.publisher_controller_right = rospy.Publisher('/external_references_for_right_arm', ExternalReference,queue_size=1)
        self.publisher_rigth_arm = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command',
                                                   Float64MultiArray, queue_size=1)
        self.publisher_teleport = rospy.Publisher('/canopies_simulator/moving_base/teleport',
                                                   Pose, queue_size=1)
        self.listener = tf.TransformListener()
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

        self.clip_rotations = True
        self.random_start = True



    def main(self):

        if self.random_start:
            self.random_init()
            self.reset()
            rospy.sleep(5)
        else:
            self.reset()
        self.torso_controller(0.085)
        rospy.sleep(5)

        ## ---------------- SIM LOOP ----------------
        rospy.loginfo(f'ROS workspace is ready for task {self.task}. Press the lateral button to start!')

        while not self.recording:
            rospy.sleep(0.1)

        ee_pos_msg = ExternalReference()
        offset = self.init_pos - copy.deepcopy(self.vr_pos)
        offset_rot = tf.transformations.quaternion_inverse(copy.deepcopy(self.vr_rot))

        self.sim_step=0
        while not rospy.is_shutdown():

            #checks
            target_id, target_dist = self.grape_remove_check(self.ee_pos)

            if self.sim_step%50==0:
                give_instructions(target_id, target_dist)


            if self.resetting:
                print('resetting...')
                self.reset()

            elif self.discard:
                rospy.loginfo('Trajectory discarded')
                self.traj_buffer.reset()

            elif self.save:
                if self.traj_buffer.size()>10:
                    self.traj_buffer.grapes_positions = copy.deepcopy(list(self.grapes_pos))
                    self.traj_buffer.show_current_status()
                    save_path = self.traj_buffer.save_trajectory(f'{str(round(datetime.datetime.now().timestamp()))}.pkl')
                    rospy.loginfo(f'Trajectory saved in {save_path}')
                    self.traj_buffer.reset()
                    self.save = False

            elif self.block:
                self.velocity_msg_right.data = [0.0] * 8
                self.publisher_rigth_arm.publish(self.velocity_msg_right)

            else:

                target_pos = offset + self.vr_pos
                target_or = tf.transformations.quaternion_multiply(self.vr_rot, offset_rot)

                if self.clip_rotations:
                    target_euler = np.clip(
                        tf.transformations.euler_from_quaternion(target_or),
                        a_min=-np.pi/3,
                        a_max=np.pi/3
                    )
                    target_euler *= 0.8
                target_or = tf.transformations.quaternion_from_euler(*target_euler)

                if self.vr_mirroring:
                    target_or *= [1., 1., -1., -1.]

                if self.recording:
                    self.traj_buffer.target_orientations['value'].append(list(target_or))
                    self.traj_buffer.target_positions['value'].append(list(target_pos))
                    self.traj_buffer.target_positions['time'].append(rospy.get_time())
                    self.traj_buffer.target_orientations['time'].append(rospy.get_time())

                ee_pos_msg.position = Point(x=target_pos[0], y=target_pos[1], z=target_pos[2])
                ee_pos_msg.orientation = Quaternion(x=target_or[0], y=target_or[1], z=target_or[2], w=target_or[3])
                self.publisher_controller_right.publish(ee_pos_msg)

            self.sim_step+=1
            self.control_loop_rate.sleep()

    ##  -----  CALLBACKS  ------------------------------------------------

    def callback_vr_position_right_arm(self, vr_pose):
        t = rospy.get_time()
        mirror = -1. if self.vr_mirroring else 1.

        self.vr_pos = np.array(
            [vr_pose.pose.position.x,
            mirror*vr_pose.pose.position.y,
            vr_pose.pose.position.z]
        )
        self.vr_rot = np.array(
            [vr_pose.pose.orientation.x,
            vr_pose.pose.orientation.y,
            vr_pose.pose.orientation.z,
            vr_pose.pose.orientation.w]
        )
        #if self.vr_mirroring:
        #self.vr_rot = tf.transformations.quaternion_multiply(self.vr_rot, [0., 0., 1., 0.])


    def callback_velocity_commands(self, msg_vel):
        if self.block:
            self.velocity_msg_right.data = [0.0] * 8
        else:
            for i, name in enumerate(self.names_right):
                index_in_msg = self.names_right.index(name)
                self.velocity_msg_right.data[index_in_msg] = msg_vel.data[i] * self.k_v
        self.publisher_rigth_arm.publish(self.velocity_msg_right)

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

    ## OTHER METHODS
    def grape_remove_check(self, ee_pos):
        dists_3d = self.grapes_pos - np.array(ee_pos)
        dists = np.linalg.norm(dists_3d, axis=-1)
        target_id = self.grapes_idx[np.argmin(dists)]
        threshold = (dists < self.threshold)*1.
        for i in np.nonzero(threshold)[0]:
            #if self.grapes_idx[i] not in self.removed_grapes:
            rospy.loginfo(f'\n{self.grapes_idx[i]} grape removed')
                #self.simulator_remove_grape_bunch(int(self.grapes_idx[i]))
                #self.removed_grapes.append(self.grapes_idx[i])

        return target_id, dists_3d[np.argmin(dists)]

    def mobile_base_controller(self, v, u):
        twist = Twist()
        mirror = -1 if self.vr_mirroring else 1
        twist.linear.x = v * mirror
        twist.angular.z = u * mirror
        self.publisher_mobile_base.publish(twist)

    def torso_controller(self, h):
        """
        Moves the torso joint until a specif height (considering head_2_link wrt base_frame as reference)
        """
        x_0, _ = self.get_transform(target_frame='base_footprint', source_frame='head_2_link')
        e = 1.
        while abs(e)>0.001:
            self.velocity_msg_right.data = [0.0] * 7 + [0.1 * np.sign(e)]
            rospy.sleep(0.01)
            x, _ = self.get_transform(target_frame='base_footprint', source_frame='head_2_link')
            z = x[2] - x_0[2]
            e = h-z
        self.velocity_msg_right.data = [0.0] * 8
        self.publisher_joint_commands.publish(self.velocity_msg_right)
        return x


    def random_init(self):
        """
        Initialize the robot in a pseudo-random position
        """

        if self.vr_mirroring:
            x_t, y_t, z_t = random.uniform(-0.3, 0.1), random.uniform(-0.1, 0.5), 0.24 + random.uniform(-0.0, 0.0)
            th_t = np.pi
        else:
            x_t, y_t, z_t = random.uniform(-0.1, 0.6), random.uniform(-0.5, 0.1), 0.24
            th_t = 0 #np.arctan2(0., x_t)


        (x_g, y_g, z_g) = (0.8866932392120361, -1.568265438079834, 1.7083934545516968)
        #(8.601644515991211, -1.2150168418884277, 1.6853939294815063)
        #(x_g, y_g, z_g), _ = self.get_transform(target_frame='map', source_frame=f'Bunch_27')
        pose = Pose()
        pose.position = Point(x_g - x_t, y_g - y_t, z_t)
        pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0., 0., th_t))
        self.publisher_teleport.publish(pose)

        print(f'Teleported to {(x_t, y_t, z_t)}')

    def reset(self):
        ee_pos_msg = ExternalReference()
        ee_pos_msg.position = Point(x=self.init_pos[0], y=self.init_pos[1], z=self.init_pos[2])
        ee_pos_msg.orientation = Quaternion(x=self.init_rot[0], y=self.init_rot[1], z=self.init_rot[2],
                                            w=self.init_rot[3])
        self.publisher_controller_right.publish(ee_pos_msg)

    def simulator_remove_grape_bunch(self, id_: int):
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

def give_instructions(target_id,target_dist):
    dx, dy, dz = target_dist
    rospy.loginfo(f'\n{target_id}: {target_dist} ({np.linalg.norm(target_dist)}):')
    rospy.loginfo(f' - X: {"Forward" if dx >0.0 else "Backward"} by {abs(dx)}')
    rospy.loginfo(f' - Y: {"Left" if dx >0. else "Right"} by {abs(dy)}')
    rospy.loginfo(f' - Z: {"Up" if dx >0. else "Down"} by {abs(dz)}')


if __name__ == "__main__":
    try:
        node = VRCommands()
        node.main()
    except rospy.ROSInterruptException:
        pass


