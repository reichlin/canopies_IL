#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped, TwistStamped, Twist
from std_msgs.msg import Float64MultiArray
from farming_robot_control_msgs.msg import ExternalReference
from canopies_simulator.msg import BoundBoxArray
from quest2ros.msg import OVR2ROSInputs

import numpy as np
import time
import datetime



vr_pos = np.array([0.0, 0.0, 0.0])
vr_rot = np.array([0.0, 0.0, 0.0])
vr_pos_vel = np.array([0.0, 0.0, 0.0])
vr_rot_vel = np.array([0.0, 0.0, 0.0])

# Data to record
data_pos = []
data_or = []
data_vel = []
data_joints_pos = []
data_joints_vel = []

term = False
block = 0.0


def callback_vr_velocity_right_arm(vr_twist):
    global vr_pos_vel, vr_rot_vel
    vr_pos_vel[0], vr_pos_vel[1], vr_pos_vel[2] = vr_twist.linear.x, vr_twist.linear.y, vr_twist.linear.z
    #vr_rot_vel = vr_twist.angular


def callback_vr_position_right_arm(vr_pose):
    global vr_pos, vr_rot
    vr_pos = vr_pose.pose.position
    vr_rot = vr_pose.pose.orientation


def callback_vr_inputs_right_arm(vr_inputs):
    global term, block
    term = vr_inputs.button_upper or vr_inputs.button_lower
    block = vr_inputs.press_index


def callback_joint_state(joints_msg):
    global data_joints_pos, data_joints_vel
    data_joints_pos.append(np.expand_dims(np.array(joints_msg.actual.positions), 0))
    data_joints_vel.append(np.expand_dims(np.array(joints_msg.actual.velocities), 0))


def callback_end_effector_state(end_effector_msg):
    global data_pos, data_or
    data_pos.append(np.expand_dims(np.array(end_effector_msg.pose.position), 0))
    data_or.append(np.expand_dims(np.array(end_effector_msg.pose.orientation), 0))


def main():

    global data_joints_pos, data_joints_vel, data_pos, data_or, data_vel

    rospy.init_node("vr_commands", anonymous=True)

    rospy.sleep(2)

    publisher_controller_right = rospy.Publisher('/external_references_for_right_arm', ExternalReference, queue_size=1)

    rospy.Subscriber('/q2r_right_hand_pose', PoseStamped, callback_vr_position_right_arm)
    rospy.Subscriber('/q2r_right_hand_twist', Twist, callback_vr_velocity_right_arm)
    rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, callback_vr_inputs_right_arm)

    rospy.Subscriber('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, callback_joint_state)
    rospy.Subscriber('/task_2_value', PoseStamped, callback_end_effector_state)

    grapes_info = rospy.wait_for_message('/canopies_simulator/grape_boxes', BoundBoxArray, timeout=5)

    data_external_objs = []
    for grape in grapes_info.boxes:
        data_external_objs.append(np.array([[[grape.xmin, grape.xmax], [grape.ymin, grape.ymax], [grape.zmin, grape.zmax]]]))
    data_external_objs = np.concatenate(data_external_objs, 0)
    print(data_external_objs.shape)



    control_loop_rate = rospy.Rate(100)  # 10Hz

    print("starting ...")

    # velocity_msg = Float64MultiArray()
    # velocity_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Initialize arm in a standard position
    ee_pos_msg = ExternalReference()
    ee_pos_msg.position = Point(x=0.093, y=0.1, z=-0.0728)
    ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    publisher_controller_right.publish(ee_pos_msg)
    rospy.sleep(2)

    pos = np.array([0.093, 0.1, -0.0728])

    print("ready ...")

    t = time.time()
    while not rospy.is_shutdown():

        # vr_inputs = rospy.wait_for_message('/q2r_right_hand_inputs', OVR2ROSInputs, timeout=5)

        if term:
            break

        if block == 1.0:
            pass
        else:

            # TODO: still not sure if 100Hz is the best frequency (send array of waypoints?)
            # TODO: signs are wrong for some directions
            # TODO: now all actions are saved, definitely too much, on the other hand maybe I don't even need them

            # joints_info = rospy.wait_for_message('/canopies_simulator/arm_right_controller/state', JointTrajectoryControllerState, timeout=5)
            # data_joints_pos.append(np.expand_dims(np.array(joints_info.actual.positions), 0))
            # data_joints_vel.append(np.expand_dims(np.array(joints_info.actual.velocities), 0))
            #
            # ee_info = rospy.wait_for_message('/task_2_value', PoseStamped, timeout=5)
            # data_pos.append(np.expand_dims(np.array(ee_info.pose.position), 0))
            # data_or.append(np.expand_dims(np.array(ee_info.pose.orientation), 0))

            action = vr_pos_vel * (time.time() - t)

            data_vel.append(np.expand_dims(action, 0))

            pos += action
            t = time.time()

            ee_pos_msg.position = Point(x=pos[0], y=pos[1], z=pos[2])
            ee_pos_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            publisher_controller_right.publish(ee_pos_msg)

            control_loop_rate.sleep()

    data_joints_pos = np.concatenate(data_joints_pos, 0)
    data_joints_vel = np.concatenate(data_joints_vel, 0)
    data_pos = np.concatenate(data_pos, 0)
    data_or = np.concatenate(data_or, 0)
    data_vel = np.concatenate(data_vel, 0)

    print("saving data ...")

    trj_n = int(datetime.datetime.now().timestamp())
    np.savez("/home/alfredo/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data/trj_n="+str(trj_n)+".npz",
             data_joints_pos,
             data_joints_vel,
             data_pos,
             data_or,
             data_vel,
             data_external_objs)





if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


