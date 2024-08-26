#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
import tf

class FrameListener:
    def __init__(self):
        self.publisher_mobile_base = rospy.Publisher('/direct_kinematics/rigth_arm_end_effector', PoseStamped, queue_size=1)

    def listen(self, target_frame, source_frame):
        rospy.init_node('frame_listener', anonymous=True)
        listener = tf.TransformListener()
        msg = PoseStamped()
        msg.header.frame_id = source_frame
        listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(5.0))
        (last_trans, last_rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

        rate = rospy.Rate(50.0)  # 10 Hz

        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(1.0))
                t = rospy.get_time()
                (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

                dist = np.sum(np.abs(np.subtract(trans, last_trans)))

                if dist > 0.00001:
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = trans
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = rot
                    msg.header.stamp.secs = int(t)
                    msg.header.stamp.nsecs = int((t - int(t)) * 10**9)
                    self.publisher_mobile_base.publish(msg)
                last_trans, last_rot = trans, rot
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                continue

            rate.sleep()

if __name__ == '__main__':
    listener = FrameListener()
    target_frame = 'base_footprint'
    source_frame = f'inner_finger_1_right_link'
    listener.listen(target_frame, source_frame)
