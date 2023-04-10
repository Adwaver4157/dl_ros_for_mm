import argparse

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32


class BaseAgent:
    def __init__(self, model=None):
        self.model = model
        self.base_pub = rospy.Publisher("/mss/m2s/base_pub", Twist, queue_size=1)
        self.pose_pub_2 = rospy.Publisher(
            "/controller/pose_c2", PoseStamped, queue_size=1
        )
        self.trigger_pub_1 = rospy.Publisher(
            "/controller/trigger_c1", Float32, queue_size=1
        )
        self.trigger_pub_2 = rospy.Publisher(
            "/controller/trigger_c2", Float32, queue_size=1
        )

        # while not rospy.is_shutdown():
        #     self.step()

        # rospy.spin()

    def step(self):
        base_cmd = Twist()
        base_cmd.linear.x = 0.0
        base_cmd.angular.z = 0.0
        self.base_pub.publish(base_cmd)

        pose_cmd = PoseStamped()
        pose_cmd.header.stamp = rospy.Time.now()
        pose_cmd.header.frame_id = "base_link"
        pose_cmd.pose.position.x = 0.0
        pose_cmd.pose.position.y = 0.0
        pose_cmd.pose.position.z = 0.0
        pose_cmd.pose.orientation.x = 0.0
        pose_cmd.pose.orientation.y = 0.0
        pose_cmd.pose.orientation.z = 0.0
        pose_cmd.pose.orientation.w = 0.0
        self.pose_pub_2.publish(pose_cmd)

        trigger_cmd_1 = Float32()
        trigger_cmd_1.data = 0.0
        self.trigger_pub_1.publish(trigger_cmd_1)

        trigger_cmd_2 = Float32()
        trigger_cmd_2.data = 0.0
        self.trigger_pub_2.publish(trigger_cmd_2)


if __name__ == "__main__":
    rospy.init_node("base_agent")
    BaseAgent()
