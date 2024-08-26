#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Quaternion, PoseStamped,Pose
from std_msgs.msg import Float64MultiArray
import random
from farming_robot_control_msgs.msg import ExternalReference
from canopies_simulator.msg import BoundBoxArray
import numpy as np
import torch
import os
import tf

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('imitation_learning')
module_path = os.path.join(package_path, '/src/utils')
sys.path.append(module_path)
from utils import Buffer, simulator_remove_grape_bunch, check_workspace
from utils_rl import Agent

class ImitationNode:

    def __init__(self):

        # get rosparams
        self.names_right = rospy.get_param('/arm_right_joints')
        self.load_dir = os.path.join(package_path, rospy.get_param('/folders/load'))
        self.save_dir = os.path.join(package_path, rospy.get_param('/folders/result'))
        self.recording = rospy.get_param('rec')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.threshold = rospy.get_param('/threshold')
        self.n_frames = rospy.get_param('/agent_hp/n_frames')
        self.task = rospy.get_param('task')

        #init variables
        self.init_pose = np.array(rospy.get_param('init_pose'))
        self.joint_pos, self.joint_vel = np.zeros(7), np.zeros(7)
        self.torso_height = rospy.get_param('torso_height')
        self.ee_pos_msg = ExternalReference()
        self.velocity_msg_right = Float64MultiArray()
        self.velocity_msg_right.data = [0.0] * 8
        self.k_v = rospy.get_param('/gains/k_v')
        self.saved = False
        self.random_start = False

        # setup the recorder
        self.traj_buffer = Buffer(self.save_dir)

        # ROS stuff
        rospy.init_node("high_level_controller", anonymous=True)

        rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', self.names_right+['torso_lift_joint'])
        rospy.set_param('canopies_simulator/joint_states/rate', 100)
        self.tf_listener = tf.TransformListener()

        rospy.Subscriber('/canopies_simulator/joint_states', JointState, self.callback_joint_state)
        rospy.Subscriber('/canopies_simulator/grape_boxes', BoundBoxArray, self.callback_grapes, queue_size=1)
        rospy.Subscriber('/direct_kinematics/rigth_arm_end_effector', PoseStamped, self.callback_end_effector, queue_size=1)
        rospy.Subscriber('/arm_right_forward_velocity_controller/command', Float64MultiArray, self.callback_velocity_commands, queue_size=1)

        self.publisher_joint_commands = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command',Float64MultiArray, queue_size=1)
        self.publisher_teleport = rospy.Publisher('/canopies_simulator/moving_base/teleport',Pose, queue_size=1)
        self.publisher_ee_commands = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)

        rate = rospy.get_param('rates/testing')
        self.control_loop_rate = rospy.Rate(rate)
        self.rate=rate
        rospy.sleep(2)

        # init agent and load param
        agent_hp = rospy.get_param('agent_hp')
        self.k_ca = rospy.get_param('gains/k_ca')
        self.collision_avoidance = agent_hp['collision_avoidance']
        self.agent = Agent(
            input_size= agent_hp['input_dim'], #rospy.get_param('/agent_hp/input_dim')*self.n_frames,
            goal_size=agent_hp['goal_dim'],
            action_size=(3,4),
            N_gaussians=agent_hp['n_gaussians'],
            stable=agent_hp['stable'],
            sigma=rospy.get_param('agent_hp/sigma'),
            device=self.device
        ).to(self.device)
        models_folder = os.path.join(package_path, self.load_dir, f'grasping_J_only_{rate}hz_{self.task}')
        self.agent.load_models(models_folder)
        self.agent.eval()

        # init the grapes
        self.removed_grapes = []
        #self.grapes_pos, self.grapes_idx, self.removed_grapes = []*3 #self.register_grapes()


    def main(self):


        ## ====================================== SETUP =====================================

        #locate the robot
        if self.random_start:
            self.random_init()
            self.reset()
            #rospy.sleep(5)
        else:
            self.reset()
        rospy.sleep(1)

        #let the user choose the goal position
        target_pos, target_rot = self.init_pose[:3], self.init_pose[3:]
        goal_pos, goal_id = self.get_goal()
        self.goal = torch.tensor(goal_pos[:2]).float().unsqueeze(0).to(self.device)
        self.head_height_controller(goal_pos[-1] + 0.1 - self.torso_height)
        rospy.sleep(1)

        # collect the position of the obstacles in the latent space
        obs_pos, obs_idx = self.register_obstacles(goal_id)
        obs_tsr = torch.from_numpy(self.joint_pos).float().unsqueeze(0).to(self.device)
        z_obs = ((self.agent.encoder(obs_tsr).detach().cpu().squeeze() +
                  torch.from_numpy(obs_pos).float() -
                  torch.tensor(self.ee_pos))).float()
        sigma = 0.5**2
        print('\nObstacles:')
        for i, p in zip(obs_idx, obs_pos):
            print(f'{i}: {p.tolist()}')

        ## ====================================== SIM LOOP ========================================================

        self.sim_step = 0
        rospy.loginfo(f"Starting ... agent{' stable ' if self.agent.stable else ' '}aiming to {goal_pos} of id {goal_id}")

        while not rospy.is_shutdown():

            if goal_id in self.removed_grapes:
                self.block()

            else:
                obs_tsr = torch.from_numpy(self.joint_pos).float().unsqueeze(0).to(self.device)

                # get the new displacement for the viapoint (position) and the new orientation
                action = self.agent.select_action(
                    obs_tsr, self.goal
                ).detach().cpu().squeeze().numpy()

                if self.collision_avoidance:
                    action_CA = np.zeros(7)
                    act, p = self.agent.compute_CA(
                        obs_tsr, z_obs.to(self.device), sigma=sigma
                    )
                    action[:3] += act[:3].detach().cpu().squeeze().numpy()*self.k_ca
                    #print('\n', p.item(), ' - ',act)


                # compute and publish the new commanded pose
                d_target_pos, target_rot = action[:3], action[3:]
                target_pos += d_target_pos
                #target_rot = data_orientations[self.sim_step]
                #target_pos += data_dpositions[self.sim_step]
                self.send_viapoint(target_pos, target_rot)
                t = rospy.get_time()

                #remove a grape if necessary
                self.grape_revove_check(self.ee_pos)

                if self.sim_step % 50==0:
                    rospy.loginfo(f"{self.sim_step} Target: {target_pos} - {target_rot}")

                if self.recording:
                    self.traj_buffer.target_positions['value'].append(list(target_pos))
                    self.traj_buffer.target_orientations['value'].append(list(target_rot))
                    self.traj_buffer.command_positions['value'].append(list(d_target_pos))
                    self.traj_buffer.target_positions['time'].append(t)
                    self.traj_buffer.target_orientations['time'].append(t)
                    self.traj_buffer.command_positions['time'].append(t)

                    if (not self.removed_grapes==[] or self.sim_step == 1000) and not self.saved:
                        self.traj_buffer.grapes_positions.append(self.goal.detach().cpu().unsqueeze(0).numpy())
                        self.traj_buffer.show_current_status()
                        self.traj_buffer.save_trajectory(f'{self.task}.pkl')
                        self.saved = True

            self.sim_step += 1
            self.control_loop_rate.sleep()




    ## ====================================== CALLBACKS ==========================================================

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


    ## ====================================== OTHER METHODS ==========================================================

    def grape_revove_check(self, ee_pos):
        """
        Checks for bunches to remove, if under a certain threshold in respect to the distance
        """
        dists = np.linalg.norm(np.array(ee_pos) - np.array(self.grapes_pos), axis=-1)
        threshold = (dists < self.threshold)*1.
        for i in np.nonzero(threshold)[0]:
            if self.grapes_idx[i] not in self.removed_grapes:
                simulator_remove_grape_bunch(int(self.grapes_idx[i]))
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

    def register_obstacles(self, goal_id):
        """
        Returns all obstacles baricentrum.
        """
        obs_pos = []
        idxs = []
        grapes_data = rospy.wait_for_message('/canopies_simulator/grape_boxes', BoundBoxArray, timeout=None)
        for box in grapes_data.boxes:
            if not box.index == goal_id:
                obs_pos.append([
                    (box.xmax+box.xmin)/2,
                    (box.ymax+box.ymin)/2,
                    (box.zmax+box.zmin)/2
                ])
                idxs.append(box.index)
        return np.array(obs_pos), idxs


    def get_goal(self):
        """
        Asks to the user to choose goal grape to grape, returning the id and the goal position
        """
        located_correctly = False
        workspace = rospy.get_param('workspace_cylinder')

        while not located_correctly:

            input(f"\n{'-'*50}\nLocate your self in the simulator. Once done, press enter to continue.")

            available_ids = {
                g_id: (g_pos, np.linalg.norm(np.subtract(g_pos[:2], self.ee_pos[:2]), axis=-1))
                for g_id, g_pos in zip(self.grapes_idx, self.grapes_pos)
                if check_workspace(g_pos, c=workspace['centre'],
                                   r_bounds=tuple(map(float,workspace['radius_bounds'])),
                                   theta_bounds=tuple(map(float,workspace['theta_bounds'])),
                                   h=workspace['height'], idx=g_id)
            }
            if len(available_ids) == 0:
                rospy.logerr(f"No grape reachable. Please relocate again.")
            else:
                located_correctly = True

        print("\nThe grapes reachable from this position are the following:")
        for idx, (pos, dist) in available_ids.items():
            print(f' - {idx} at distance {round(dist, 3)}m ({pos}) from the end effector')

        try:
            goal_id = int(input("Input the id of the grape to collect?: "))
        except:
            goal_id = None

        while not goal_id in available_ids:
            try:
                goal_id = int(input("id not valid. Please enter a valid id (from the list): "))
            except:
                pass

        return available_ids[goal_id][0], goal_id

    def head_height_controller(self, h):
        """
        Reach the head height wrt base_footprint lifting the torso lift joint
        """
        e, z = 1., 0.
        start = rospy.get_time()
        while abs(e)>0.001:
            self.velocity_msg_right.data = [0.0] * 7 + [0.1 * np.sign(e)]
            rospy.sleep(0.01)
            (_, _, z), _ = self.get_transform(target_frame='base_footprint', source_frame='head_2_link')
            e = h - z
            if rospy.get_time() - start > 5.0:
                break
        self.velocity_msg_right.data = [0.0] * 8
        self.publisher_joint_commands.publish(self.velocity_msg_right)
        return z

    def reset(self):
        """
        Commands to the controller the init pose
        """
        ee_pos_msg = ExternalReference()
        ee_pos_msg.position = Point(x=self.init_pose[0], y=self.init_pose[1], z=self.init_pose[2])
        ee_pos_msg.orientation = Quaternion(x=self.init_pose[3],
                                            y=self.init_pose[4],
                                            z=self.init_pose[5],
                                            w=self.init_pose[6])
        self.publisher_ee_commands.publish(ee_pos_msg)


    def send_viapoint(self, position, orientation):
        """
        INPUT: viapoint postion (3D) and orientation (quaternion)
        """
        self.ee_pos_msg.position = Point(x=position[0], y=position[1], z=position[2])
        self.ee_pos_msg.orientation = Quaternion(x=orientation[0],y=orientation[1],z=orientation[2],w=orientation[3])
        self.publisher_ee_commands.publish(self.ee_pos_msg)

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
        pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0., 0., 0.))
        self.publisher_teleport.publish(pose)
        rospy.loginfo(f'Teleported to {target_pos}')
        return 27

    def block(self):
        self.velocity_msg_right.data = [0.0] * 8
        self.publisher_joint_commands.publish(self.velocity_msg_right)



if __name__ == "__main__":
    try:
        node = ImitationNode()
        node.main()
    except rospy.ROSInterruptException:
        pass
