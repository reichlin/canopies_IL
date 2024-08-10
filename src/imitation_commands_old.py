#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Quaternion, PoseStamped,Pose
from std_msgs.msg import Float64MultiArray
import random
from farming_robot_control_msgs.msg import ExternalReference
from canopies_simulator.msg import BoundBoxArray
import pickle
import numpy as np
import torch
import os
import copy
import time
from canopies_simulator.srv import Simulator
import tf
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path, '/src/utils')
sys.path.append(module_path)
from utils import Buffer
from utils_rl import Agent

class ImitationNode:

    def __init__(self):

        # get rosparams
        self.names_right = rospy.get_param('/arm_right_joints')
        self.load_dir = os.path.join(package_path, rospy.get_param('/folders/load'))
        self.save_dir = os.path.join(package_path, rospy.get_param('/folders/result'))
        self.recording = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.threshold = rospy.get_param('/threshold')
        self.n_frames = rospy.get_param('/agent_hp/n_frames')

        #init variables
        self.init_pos = np.array(rospy.get_param('ee_init_pos'))
        self.init_rot = np.array(rospy.get_param('ee_init_or'))
        self.joint_pos, self.joint_vel = np.zeros(7), np.zeros(7)
        self.ee_pos_msg = ExternalReference()
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0] * 8
        self.k_v = rospy.get_param('/gains/k_v')
        self.saved = False

        # setup the recorder
        self.traj_buffer = Buffer(self.save_dir)

        # ROS stuff
        rospy.init_node("high_level_controller", anonymous=True)

        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right+['torso_lift_joint'])
        rospy.set_param('canopies_simulator/joint_states/rate', 100)
        rospy.Subscriber('/canopies_simulator/joint_states', JointState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes, queue_size=1)
        self.publisher_joint_commands = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command',Float64MultiArray, queue_size=1)
        self.publisher_teleport = rospy.Publisher('/canopies_simulator/moving_base/teleport',Pose, queue_size=1)
        self.publisher_ee_commands = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)
        rospy.Subscriber('/direct_kinematics/rigth_arm_end_effector', PoseStamped, self.callback_end_effector, queue_size=1)
        rospy.Subscriber('/arm_right_forward_velocity_controller/command', Float64MultiArray, self.callback_velocity_commands, queue_size=1)


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
            action_size=(3,4),
            N_gaussians=25,
            stable=self.stable, # if rospy.get_param('agent_name') else False,
            device=self.device
        ).to(self.device)

        models_folder = os.path.join(package_path, self.load_dir, 'grasping_J_only_50hz_grasp_best')
        self.agent.policy.load_model(os.path.join(models_folder, 'model_policy.pth'))
        self.agent.encoder.load_model(os.path.join(models_folder, 'model_encoder.pth'))
        self.agent.MDN.load_model(os.path.join(models_folder, 'model_mdn.pth'))
        self.agent.eval()

        # init the grapes
        self.grapes_pos, self.grapes_idx, self.removed_grapes = self.register_grapes()

        self.random_start = True

    def main(self):

        if self.random_start:
            self.random_init()
            self.reset()
            rospy.sleep(5)
        else:
            self.reset()
        self.torso_controller(0.085)

        target_pos, target_rot = np.array(self.init_pos), np.array(self.init_rot)
        goal_pos, goal_id = self.get_goal()
        self.goal = torch.tensor(goal_pos).float().unsqueeze(0).to(self.device)

        ## ====================================== SIM LOOP ======================================

        self.recording = rospy.get_param('rec')
        self.sim_step = 0
        rospy.loginfo(f"Starting ... agent{' stable ' if self.stable else ' '}aiming to {goal_pos} of id {goal_id}")

        while not rospy.is_shutdown():
            obs_tsr = torch.from_numpy(self.joint_pos).float().unsqueeze(0).to(self.device)
            new_target = self.agent.select_action(
                obs_tsr, self.goal,
                torch.from_numpy(target_pos).float().unsqueeze(0).to(self.agent.device)
            ).detach().cpu().squeeze().numpy()
            target_pos = new_target[:3]
            target_rot = new_target[3:]

            # publish a new commanded pos
            self.send_viapoint(target_pos, target_rot)

            if self.sim_step%50==0:
                rospy.loginfo(f"{self.sim_step} Target: {target_pos} - {target_rot}")

            if self.recording:
                self.traj_buffer.target_positions['value'].append(list(target_pos))
                self.traj_buffer.target_orientations['value'].append(list(target_rot))
                self.traj_buffer.target_positions['time'].append(rospy.get_time())
                self.traj_buffer.target_orientations['time'].append(rospy.get_time())

            #remove a grape if necessary
            self.grape_revove_check(self.ee_pos)

            # rec variables
            if self.recording and (not self.removed_grapes==[] or self.sim_step == 1000) and not self.saved:
                self.traj_buffer.grapes_positions.append(self.goal.detach().cpu().unsqueeze(0).numpy())
                self.traj_buffer.show_current_status()
                self.traj_buffer.save_trajectory(f'{self.task}.pkl')
                self.saved = True

            self.sim_step += 1
            self.control_loop_rate.sleep()



    ## ====================================== CALLBACKS =================================================

    def callback_joint_state(self, joints_msg):
        t = rospy.get_time()

        for name in self.names_right:
            index_in_joint_state = joints_msg.name.index(name)
            index_in_msg = self.names_right.index(name)
            self.joint_pos[index_in_msg] = joints_msg.position[index_in_joint_state]
            self.joint_vel[index_in_msg] = joints_msg.velocity[index_in_joint_state]

        if self.recording:
            self.traj_buffer.joint_positions['value'].append(self.joint_pos)
            self.traj_buffer.joint_velocities['value'].append(self.joint_vel)
            self.traj_buffer.joint_positions['time'].append(t)
            self.traj_buffer.joint_velocities['time'].append(t)

    def callback_velocity_commands(self, msg_vel):
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

    def callback_end_effector(self, pose_msg):
        t = pose_msg.header.stamp.secs + pose_msg.header.stamp.nsecs/(10**9)
        self.ee_pos = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        self.ee_rot = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]
        if self.recording:
            self.traj_buffer.cartesian_positions['value'].append(self.ee_pos)
            self.traj_buffer.cartesian_orientations['value'].append(self.ee_rot)
            self.traj_buffer.cartesian_positions['time'].append(t)
            self.traj_buffer.cartesian_orientations['time'].append(t)


    ## ----------------- OTHER METHODS -----------------

    def simulator_remove_grape_bunch(self, id_: int):
        """
        Removes the bunch corresponding to the input id
        """
        rospy.wait_for_service('/simulator')
        cmd = rospy.ServiceProxy('/simulator', Simulator)
        cmd("RemoveGrapeBunch", id_, False, "")


    def grape_revove_check(self, ee_pos):
        """
        Checks for bunches to remove, if under a certain threshold in respect to the distance
        """
        dists = np.linalg.norm(np.array(ee_pos) - np.array(self.grapes_pos), axis=-1)
        threshold = (dists < self.threshold)*1.
        for i in np.nonzero(threshold)[0]:
            if self.grapes_idx[i] not in self.removed_grapes:
                self.simulator_remove_grape_bunch(int(self.grapes_idx[i]))
                rospy.loginfo(f'\n{self.grapes_idx[i]} grape removed')
                self.removed_grapes.append(self.grapes_idx[i])


    def get_transform(self, target_frame, source_frame):
        """
        Returns the translation and orientation of the target frame wrt the source_frame
        """
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            return None, None

    def register_grapes(self):
        """
        Returns the available bunches positions.
        """
        grapes_pos, grapes_idx = [], []
        grapes_data = rospy.wait_for_message('/canopies_simulator/grape_boxes', BoundBoxArray, timeout=None)
        for box in grapes_data.boxes:
            grapes_idx.append(box.index)
            g_pos, _ = self.get_transform(target_frame='base_footprint', source_frame=f'Bunch_{box.index}')
            g_pos = np.add(g_pos, [0.,0.,-0.1])
            grapes_pos.append(g_pos)
        grapes_pos = np.array(grapes_pos)
        return grapes_pos, grapes_idx, []

    def get_goal(self):
        """
        Asks to the user to choose goal grape to grape, returning the id and the goal position
        """
        input(f"\n{'-'*50}\nLocate your self in the simulator. Once done, press enter to continue.")
        available_ids = {}
        print("\nThe grapes reachable from this position are the following:")
        for g_id, g_pos in zip(self.grapes_idx, self.grapes_pos):
            dist = np.linalg.norm(np.subtract(g_pos, self.ee_pos), axis=-1)
            if dist<1.0:
                print(f' - {g_id} at distance {round(dist, 3)}m from the end effector')
                available_ids[g_id] = g_pos
        try:
            goal_id = int(input("Input the id of the grape to collect?: "))
        except:
            goal_id = None
        while not goal_id in available_ids:
            try:
                goal_id = int(input("id not valid. Please enter a valid id (from the list): "))
            except:
                pass
        return available_ids[goal_id], goal_id

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

    def reset(self):
        ee_pos_msg = ExternalReference()
        ee_pos_msg.position = Point(x=self.init_pos[0], y=self.init_pos[1], z=self.init_pos[2])
        ee_pos_msg.orientation = Quaternion(x=self.init_rot[0],
                                            y=self.init_rot[1],
                                            z=self.init_rot[2],
                                            w=self.init_rot[3])
        self.publisher_ee_commands.publish(ee_pos_msg)


    def send_viapoint(self, position, orientation):
        """
        INPUT: viapoint postion (3D) and orientation (quaternion)
        """
        self.ee_pos_msg.position = Point(x=position[0], y=position[1], z=position[2])
        self.ee_pos_msg.orientation = Quaternion(x=orientation[0],y=orientation[1],z=orientation[2],w=orientation[3])
        self.publisher_ee_commands.publish(self.ee_pos_msg)

    def random_init(self):

        x_t, y_t, z_t = random.uniform(-0.1, 0.6), random.uniform(-0.5, 0.1), 0.24
        th_t = 0

        (x_g, y_g, z_g) = (0.8866932392120361, -1.568265438079834, 1.7083934545516968)
        #(8.601644515991211, -1.2150168418884277, 1.6853939294815063)
        #(x_g, y_g, z_g), _ = self.get_transform(target_frame='map', source_frame=f'Bunch_27')
        pose = Pose()
        pose.position = Point(x_g - x_t, y_g - y_t, z_t)
        pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0., 0., th_t))
        self.publisher_teleport.publish(pose)

        print(f'Teleported to {(x_t, y_t, z_t)}')



if __name__ == "__main__":
    try:
        node = ImitationNode(rec=True)
        node.main()
    except rospy.ROSInterruptException:
        pass
