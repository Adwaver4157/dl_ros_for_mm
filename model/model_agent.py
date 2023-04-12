#!/usr/bin/env python3

import json
import os
import sys
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from torchvision import transforms

from sensor_msgs.msg import Image, JointState

from utils.utils import convert_Image, convert_JointStates

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
    def __init__(self, model, env, config="/root/catkin_ws/src/dl_ros_for_mm/configs/config.json", device="cuda"):
        super().__init__(model=model)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),  # type: ignore
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.device = device
        self.head_rgb_msg = None
        self.hand_rgb_msg = None
        self.joint_states_msg = None

        rospy.Subscriber(
            "/hsrb/head_rgbd_sensor/rgb/image_raw",
            Image,
            self.head_rgb_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/hsrb/hand_camera/image_raw",
            Image,
            self.hand_rgb_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/hsrb/joint_states",
            JointState,
            self.joint_states_callback,
            queue_size=1,
        )
        # print(config)
        if os.path.exists(config):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError("cannot find config file")
        
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
        obs = self.get_obs()
        # print(obs)
        if obs is not None:
            # print(obs)
            head_image = obs["head_image"].to(self.device)
            hand_image = obs["hand_image"].to(self.device)
            joint_state = obs["joint_state"].to(self.device)
            base_vel, pose_trans, pose_angle, pose_action = self.model(head_image, hand_image, joint_state)
            
            base_vel = base_vel.to("cpu").detach().numpy().astype(np.float64).copy()
            print(base_vel)
            base_cmd = Twist()
            base_cmd.linear.x = base_vel[0][0]
            base_cmd.angular.z = base_vel[0][1]
            self.base_pub.publish(base_cmd)

            pose_trans = pose_trans.to("cpu").detach().numpy().astype(np.float64).copy()
            pose_euler = pose_angle.to("cpu").detach().numpy().astype(np.float64).copy()
            pose_quaternion = tf.transformations.quaternion_from_euler(
                pose_euler[0][0], pose_euler[0][1], pose_euler[0][2]
            )
            pose_cmd = PoseStamped()
            pose_cmd.header.stamp = rospy.Time.now()
            pose_cmd.header.frame_id = "base_link"
            pose_cmd.pose.position.x = pose_trans[0][0]
            pose_cmd.pose.position.y = pose_trans[0][1]
            pose_cmd.pose.position.z = pose_trans[0][2]
            pose_cmd.pose.orientation.x = pose_quaternion[0]
            pose_cmd.pose.orientation.y = pose_quaternion[1]
            pose_cmd.pose.orientation.z = pose_quaternion[2]
            pose_cmd.pose.orientation.w = pose_quaternion[3]
            self.pose_pub_2.publish(pose_cmd)

            trigger_cmd_1 = Float32()
            trigger_cmd_1.data = 0.0
            self.trigger_pub_1.publish(trigger_cmd_1)

            pose_action = pose_action.to("cpu").detach().numpy().astype(np.float64).copy()
            trigger_cmd_2 = Float32()
            trigger_cmd_2.data = pose_action[0][0]
            self.trigger_pub_2.publish(trigger_cmd_2)

    def get_obs(self):
        """get ros image and joint states and convert to torch tensor

        Returns:
            Dict of torch tensors: observations of head image, hand image, joint states
        """
        torch_data = {}
        # self.head_rgb_msg = None
        # self.hand_rgb_msg = None
        # self.joint_states_msg = None
        # print(f"head: {self.head_rgb_msg}")
        # print(f"hand: {self.hand_rgb_msg}")
        # print(f"joint: {self.joint_states_msg}")
        if (
            self.head_rgb_msg is not None
            and self.hand_rgb_msg is not None
            and self.joint_states_msg is not None
        ):
            head_rgb = convert_Image(
                self.head_rgb_msg, self.config["height"], self.config["width"]
            )
            hand_rgb = convert_Image(
                self.hand_rgb_msg, self.config["height"], self.config["width"]
            )
            joint_states = convert_JointStates(self.joint_states_msg)
            torch_data["head_image"] = (
                torch.tensor(head_rgb[0], dtype=torch.float32).flip(2).permute(2, 1, 0)
            ).unsqueeze(0)
            torch_data["hand_image"] = torch.tensor(
                hand_rgb[0], dtype=torch.float32
            ).permute(2, 1, 0).unsqueeze(0)

            if self.transform:
                torch_data["head_image"] = self.transform(torch_data["head_image"])
                torch_data["hand_image"] = self.transform(torch_data["hand_image"])

            torch_data["joint_state"] = torch.tensor(
                joint_states, dtype=torch.float32
            )

            return torch_data
        else:
            return None

    def head_rgb_callback(self, msg):
        self.head_rgb_msg = msg

    def hand_rgb_callback(self, msg):
        self.hand_rgb_msg = msg

    def joint_states_callback(self, msg):
        self.joint_states_msg = msg

if __name__ == "__main__":
    rospy.init_node("model_agent")

    config = rospy.get_param('/data/config', "/root/catkin_ws/src/dl_ros_for_mm/configs/config.json")
    device = rospy.get_param('/device', "cuda")

    device = select_device(device)
    model = BCAgent()
    model.load_state_dict(torch.load("/root/catkin_ws/src/dl_ros_for_mm/best_model.pth", map_location="cuda:0"))
    model.to(device)
    env = Env(config=config)
    agent = ModelAgent(model=model, env=env)
