#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
import moveit_commander
import pdb
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
# from kdl_wrapper import kdl_wrapper


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


def do_control_loop(positions):
    print("Move joints to positions: ", positions)

    control_loop_rate = rospy.Rate(10)  # 10Hz
    control_loop_time = 5  # Sec

    global velocity_msg
    global publisher

    now = rospy.Time.now()
    while rospy.Time.now() < now + rospy.Duration.from_sec(control_loop_time):
        joint_state = rospy.wait_for_message('canopies_simulator/joint_states', JointState, 10)
        # Calculation of new velocities for every joint
        for name in names:
            index_in_joint_state = joint_state.name.index(name)
            index_in_msg = names.index(name)
            final_pos = positions[index_in_msg]
            real_pos = joint_state.position[index_in_joint_state]
            control_velocity = 100 * (final_pos - real_pos)
            velocity_msg.data[index_in_msg] = control_velocity

        publisher.publish(velocity_msg)
        control_loop_rate.sleep()

    # Stop the joints
    velocity_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    publisher.publish(velocity_msg)



#moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("canopies_tests_full", anonymous=True)
publisher = rospy.Publisher('canopies_simulator/joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)

# if (!arm_kdl_wrapper.init("yumi_body", "yumi_link_7_l"))
# ROS_ERROR("Error initiliazing right_arm_kdl_wrapper");
# arm_kdl_wrapper.ik_solver_vel->setLambda(0.3);
# arm_kdl_wrapper.fk_solver_pos->JntToCart(arm_joint_positions, current_pose, -1);

# Test left arm
rospy.sleep(2)
init_parameter_server("left")
rospy.sleep(2)

do_control_loop([-0.78, 1.4679, -1.143, 1.7095, 0.0, 1.3898, 0.0])
do_control_loop([-1.2, 1.4679, -1.143, 1.7095, 0.0, 1.3898, 0.0])

# do_control_loop([-1.0, 0.2, 0.5, -1.0, 1.0, 1.0, 0.0])
# do_control_loop([-0.78, 1.4679, -1.143, 1.7095, 0.0, 1.3898, 0.0])
# do_control_loop([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# do_control_loop([-0.78, 1.4679, -1.143, 1.7095, 0.0, 1.3898, 0.0])

#exit()

#
# # Test rigth arm
# rospy.sleep(2)
init_parameter_server("right")
rospy.sleep(2)
do_control_loop([-1.0, 0.2, 0.5, -1.0, 1.0, 1.0, 0.0])
do_control_loop([-0.78, 1.4679, 1.143, 1.7095, 0.0, 1.3898, 0.0])
do_control_loop([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
do_control_loop([-0.78, 1.4679, 1.143, 1.7095, 0.0, 1.3898, 0.0])
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group_name = "arm_left"  # Specify the group name for your robot
group = moveit_commander.MoveGroupCommander(group_name)






# pdb.set_trace()
# ik_solver = moveit_commander.MoveGroupInterface.Planner(ik_link="arm_left_7_joint")
# joint_values = ik_solver.plan(desired_pose)
#
#

ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

init_parameter_server("left")
rospy.sleep(2)
do_control_loop([-0.78, 1.4679, -1.143, 1.7095, 0.0, 1.3898, 0.0])
pdb.set_trace()
for i in range(10):
    current_pose = group.get_current_pose().pose
    # joint_positions = group.get_current_joint_values()
    # print("Joint Positions:", joint_positions)
    # desired_pose = Pose()
    # desired_pose.position.x = current_pose.position.x
    # desired_pose.position.y = current_pose.position.y
    # desired_pose.position.z = current_pose.position.z
    # desired_pose.orientation.x = current_pose.orientation.x
    # desired_pose.orientation.y = current_pose.orientation.y
    # desired_pose.orientation.z = current_pose.orientation.z
    # desired_pose.orientation.w = current_pose.orientation.w


    # Create a service request
    ik_request = GetPositionIKRequest()
    ik_request.ik_request.group_name = "arm_left"  # Specify your group name
    ik_request.ik_request.pose_stamped.header.frame_id = "base_link"  # Specify your desired frame
    ik_request.ik_request.pose_stamped.pose.position.x = current_pose.position.x
    ik_request.ik_request.pose_stamped.pose.position.y = current_pose.position.y - 0.05
    ik_request.ik_request.pose_stamped.pose.position.z = current_pose.position.z - 0.05
    ik_request.ik_request.pose_stamped.pose.orientation.x = current_pose.orientation.x
    ik_request.ik_request.pose_stamped.pose.orientation.y = current_pose.orientation.y
    ik_request.ik_request.pose_stamped.pose.orientation.z = current_pose.orientation.z
    ik_request.ik_request.pose_stamped.pose.orientation.w = current_pose.orientation.w

    ik_response = ik_service(ik_request)

    if ik_response.error_code.val == ik_response.error_code.SUCCESS:
        joint_positions = ik_response.solution.joint_state.position
        print("Joint Positions:", joint_positions)
    else:
        print("IK calculation failed with error code:", ik_response.error_code.val)

    do_control_loop(joint_positions)

# pdb.set_trace()

# print(joint_values)