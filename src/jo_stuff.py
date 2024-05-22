#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Float64MultiArray
from farming_robot_control_msgs.msg import ExternalReference

import numpy as np


names = None
publisher = None
velocity_msg = Float64MultiArray()
velocity_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def init_parameter_server(arm_name: str):
    # Init params. Send arm names first
    global names
    names = [f"arm_{arm_name}_1_joint", f"arm_{arm_name}_2_joint", f"arm_{arm_name}_3_joint", f"arm_{arm_name}_4_joint", f"arm_{arm_name}_5_joint", f"arm_{arm_name}_6_joint", f"arm_{arm_name}_7_joint"]
    print("\nCurrent arm joints: ", names)
    rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', names)

def do_control_loop(): #positions
    # print("Move joints to positions: ", positions)

    control_loop_rate = rospy.Rate(100)  # 10Hz
    control_loop_time = 5  # Sec

    global velocity_msg
    global publisher

    # now = rospy.Time.now()
    # while rospy.Time.now() < now + rospy.Duration.from_sec(control_loop_time):
    #     joint_state_canopies = rospy.wait_for_message('canopies_simulator/joint_states', JointState, 10)
    #     joint_state_ik = rospy.wait_for_message('/joint_states', JointState, 10)
    #
    #     for name in names:
    #         index_in_joint_state = joint_state_canopies.name.index(name)
    #         index_in_msg = names.index(name)
    #         final_pos = joint_state_ik.position[index_in_msg]
    #         real_pos = joint_state_canopies.position[index_in_joint_state]
    #         control_velocity = 100 * (final_pos - real_pos)
    #         velocity_msg.data[index_in_msg] = control_velocity
    #
    #     publisher.publish(velocity_msg)
    #     control_loop_rate.sleep()
    for t in range(20):
        joint_state_canopies = rospy.wait_for_message('canopies_simulator/joint_states', JointState, 10)
        joint_vel_ik = rospy.wait_for_message('/canopies_cmd', Float64MultiArray, 10)
        # new_joint_pos = rospy.wait_for_message('/arm_right_forward_velocity_controller/command', Float64MultiArray, 10)

        for i, name in enumerate(names):
            # index_in_joint_state = joint_state_canopies.name.index(name)
            index_in_msg = names.index(name)
            # final_pos = new_joint_pos.data[i]
            # real_pos = joint_state_canopies.position[index_in_joint_state]
            # control_velocity = 100 * (final_pos - real_pos)
            print(name, joint_vel_ik.data[25+i], end=" ")
            velocity_msg.data[index_in_msg] = joint_vel_ik.data[25+i] #new_joint_pos.data[i] #control_velocity
        print()

        publisher.publish(velocity_msg)
        control_loop_rate.sleep()

    # Stop the joints
    velocity_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    publisher.publish(velocity_msg)


rospy.init_node("canopies_tests_full", anonymous=True)
publisher = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)

rospy.sleep(2)
init_parameter_server("right")

# do_control_loop([1.5, 0.2, 0.5, -1.0, 1.0, 1.0, 0.0])
# do_control_loop([-0.78, 1.4679, 1.143, 1.7095, 0.0, 1.3898, 0.0])

jo_controller_right = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)

control_loop_rate = rospy.Rate(100)  # 10Hz
control_loop_time = 10000 #120  # Sec

print("starting ...")

# print_joint_states()

control_loop_rate.sleep()
control_loop_rate.sleep()


print("resetted")

now = rospy.Time.now()
while rospy.Time.now() < now + rospy.Duration.from_sec(120):

    # joint_vel_ik = rospy.wait_for_message('/canopies_cmd', Float64MultiArray, 10)
    joint_vel_ik = rospy.wait_for_message('/arm_right_forward_velocity_controller/command', Float64MultiArray, 10)
    current_poses_ee = rospy.wait_for_message('/right_arm_direct_kinematics', PoseArray, 10)

    current_pose_ee = current_poses_ee.poses[0] # 1 is desired (not clear what it means)
    ee_pos_msg = ExternalReference()
    pos_ee = current_pose_ee.position
    rot_ee = current_pose_ee.orientation

    # starting point should be:
    # position = [0.34, -0.44, 0.50]
    # orientation = [0.55, 0.60, -0.012, 0.58]


    ee_pos_msg.position = Point(x=0.3, y=0.1, z=-0.3)
    # ee_pos_msg.position = Point(x=0.0, y=-0.0, z=0.0)
    # ee_pos_msg.position = Point(x=pos_ee.x-0.39, y=pos_ee.y+0.45, z=pos_ee.z-0.88)

    ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    # ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    # ee_pos_msg.orientation = Quaternion(x=rot_ee.x, y=rot_ee.y, z=rot_ee.z, w=rot_ee.w)

    jo_controller_right.publish(ee_pos_msg)

    for i, name in enumerate(names):
        # index_in_joint_state = joint_state_canopies.name.index(name)
        index_in_msg = names.index(name)
        # final_pos = new_joint_pos.data[i]
        # real_pos = joint_state_canopies.position[index_in_joint_state]
        # control_velocity = 100 * (final_pos - real_pos)
        print(joint_vel_ik.data[i], end=" ")
        velocity_msg.data[index_in_msg] = 10 * joint_vel_ik.data[i]
        # velocity_msg.data[index_in_msg] = joint_vel_ik.data[25 + i]
    print()
    publisher.publish(velocity_msg)
    #control_loop_rate.sleep()


    # do_control_loop()

    control_loop_rate.sleep()

velocity_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
publisher.publish(velocity_msg)

print("done first part")




now = rospy.Time.now()
while rospy.Time.now() < now + rospy.Duration.from_sec(control_loop_time):

    # joint_vel_ik = rospy.wait_for_message('/canopies_cmd', Float64MultiArray, 10)
    joint_vel_ik = rospy.wait_for_message('/arm_right_forward_velocity_controller/command', Float64MultiArray, 10)
    current_poses_ee = rospy.wait_for_message('/right_arm_direct_kinematics', PoseArray, 10)

    current_pose_ee = current_poses_ee.poses[0] # 1 is desired (not clear what it means)
    ee_pos_msg = ExternalReference()
    pos_ee = current_pose_ee.position
    rot_ee = current_pose_ee.orientation

    # starting point should be:
    # position = [0.34, -0.44, 0.50]
    # orientation = [0.55, 0.60, -0.012, 0.58]

    ee_pos_msg.position = Point(x=0.0, y=-0.0, z=0.0)
    # ee_pos_msg.position = Point(x=pos_ee.x-0.39, y=pos_ee.y+0.45, z=pos_ee.z-0.88)

    ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    # ee_pos_msg.orientation = Quaternion(x=rot_ee.x, y=rot_ee.y, z=rot_ee.z, w=rot_ee.w)

    jo_controller_right.publish(ee_pos_msg)

    for i, name in enumerate(names):
        # index_in_joint_state = joint_state_canopies.name.index(name)
        index_in_msg = names.index(name)
        # final_pos = new_joint_pos.data[i]
        # real_pos = joint_state_canopies.position[index_in_joint_state]
        # control_velocity = 100 * (final_pos - real_pos)
        print(joint_vel_ik.data[i], end=" ")
        velocity_msg.data[index_in_msg] = joint_vel_ik.data[i]
        # velocity_msg.data[index_in_msg] = joint_vel_ik.data[25 + i]
    print()
    publisher.publish(velocity_msg)
    #control_loop_rate.sleep()


    # do_control_loop()

    control_loop_rate.sleep()

velocity_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
publisher.publish(velocity_msg)







