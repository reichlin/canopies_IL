#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Float64MultiArray
from farming_robot_control_msgs.msg import ExternalReference
from quest2ros.msg import OVR2ROSInputs

import numpy as np

names_right = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
names_left = ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint"]
velocity_msg_right = Float64MultiArray()
velocity_msg_left = Float64MultiArray()
velocity_msg_right.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
velocity_msg_left.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
k_v = 20
block = 0.0


def callback_vr_inputs_right_arm(vr_inputs):
    global block
    block = vr_inputs.press_middle


def callback_right_arm(joint_vel_ik_right):

    for i, name in enumerate(names_right):
        index_in_msg = names_right.index(name)
        velocity_msg_right.data[index_in_msg] = joint_vel_ik_right.data[i] * k_v


def main():

    rospy.init_node("canopies_low_level_commands", anonymous=True)

    publisher = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)

    rospy.Subscriber('/arm_right_forward_velocity_controller/command', Float64MultiArray, callback_right_arm)

    rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, callback_vr_inputs_right_arm)

    rospy.sleep(2)

    control_loop_rate = rospy.Rate(10)  # 10Hz

    names_right = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
    names_left = ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint"]
    print("\nCurrent arm joints: ", names_right)
    rospy.set_param('canopies_simulator/joint_group_velocity_controller/joints', names_right)

    rospy.sleep(2)

    print("starting ...")


    while not rospy.is_shutdown():

        # vr_inputs = rospy.wait_for_message('/q2r_right_hand_inputs', OVR2ROSInputs, timeout=5)
        if block == 1.0:
            velocity_msg_right.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            publisher.publish(velocity_msg_right)
        else:
            publisher.publish(velocity_msg_right)
            control_loop_rate.sleep()

    velocity_msg_right.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    publisher.publish(velocity_msg_right)



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass






