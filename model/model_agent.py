import argparse

import numpy as np
import torch
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32

from model.base_agent import BaseAgent
from model.bc import BCAgent
from model.env import Env
from utils.device import select_device


class ModelAgent(BaseAgent):
    def __init__(self, model=BCAgent(), env=Env()):
        super().__init__(model=model)
        self.model = model
        self.model.eval()
        self.env = env
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

        while not rospy.is_shutdown():
            self.step()

        rospy.spin()

    def step(self):
        obs = self.env.get_obs()
        base_vel, pose_trans, pose_angle, pose_aciton = self.model(**obs)

        base_cmd = Twist()
        base_cmd.linear.x = base_vel[0]
        base_cmd.angular.z = base_vel[1]
        self.base_pub.publish(base_cmd)

        pose_euler = pose_angle.to("cpu").detach().numpy().astype(np.uint8).copy()
        pose_quaternion = tf.transformations.quaternion_from_euler(
            pose_euler[0], pose_euler[1], pose_euler[2]
        )
        pose_cmd = PoseStamped()
        pose_cmd.header.stamp = rospy.Time.now()
        pose_cmd.header.frame_id = "base_link"
        pose_cmd.pose.position.x = pose_trans[0]
        pose_cmd.pose.position.y = pose_trans[1]
        pose_cmd.pose.position.z = pose_trans[2]
        pose_cmd.pose.orientation.x = pose_quaternion[0]
        pose_cmd.pose.orientation.y = pose_quaternion[1]
        pose_cmd.pose.orientation.z = pose_quaternion[2]
        pose_cmd.pose.orientation.w = pose_quaternion[3]
        self.pose_pub_2.publish(pose_cmd)

        trigger_cmd_1 = Float32()
        trigger_cmd_1.data = 0.0
        self.trigger_pub_1.publish(trigger_cmd_1)

        trigger_cmd_2 = Float32()
        trigger_cmd_2.data = pose_aciton
        self.trigger_pub_2.publish(trigger_cmd_2)


if __name__ == "__main__":
    rospy.init_node("model_agent")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()

    device = select_device(args.device)
    model = BCAgent()
    model.load_state_dict(torch.load("best_weight.pth"))
    model.to(device)
    env = Env(config=args.config)
    agent = ModelAgent(model=model, env=env)
