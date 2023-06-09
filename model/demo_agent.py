#!/usr/bin/env python3

import json
import os
from torch.utils.data import DataLoader
import sys
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from torchvision import transforms

from sensor_msgs.msg import Image, JointState

from utils.utils import convert_Image, convert_JointStates
from utils.dataset import MyDataset

import argparse

import numpy as np
import torch
import rospy
import tf

from geometry_msgs.msg import PoseStamped, Twist, Vector3, Quaternion
from std_msgs.msg import Float32

from model.base_agent import BaseAgent
from model.bc import BCAgent
from model.env import Env
from utils.device import select_device


class DemoAgent(BaseAgent):
    def __init__(
        self,
        model=None,
        config="/root/catkin_ws/src/dl_ros_for_mm/configs/config.json",
        device="cuda",
        data=None,
    ):
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
        if data is not None:
            self.data = data
        else:
            ValueError("data is None")

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

        self.prev_pos_trans = Vector3(0, 0, 0)
        self.prev_pos_quat = Quaternion(0, 0, 0, 1)

        rate = rospy.Rate(5)
        self.index = 0
        while not rospy.is_shutdown():
            if self.index < len(self.data["base_cmd"]) - 1:
                self.step(self.index)
                self.index += 1
            else:
                break
            rate.sleep()
        trigger_cmd_2 = Float32()
        trigger_cmd_2.data = 0.0
        self.trigger_pub_2.publish(trigger_cmd_2)
        rospy.spin()

        print("Episode is done")

    def step(self, step):
        base_vel, pose_trans, pose_angle, pose_action = self.get_action(step)

        base_vel = base_vel.to("cpu").detach().numpy().astype(np.float64).copy()
        base_cmd = Twist()
        base_cmd.linear.x = base_vel[0]
        base_cmd.angular.z = base_vel[1]
        # print(base_cmd)

        # to_ros = False
        to_ros = True
        if to_ros:
            self.base_pub.publish(base_cmd)
        pose_trans = pose_trans.to("cpu").detach().numpy().astype(np.float64).copy()
        pose_euler = pose_angle.to("cpu").detach().numpy().astype(np.float64).copy()
        pose_quaternion = tf.transformations.quaternion_from_euler(
            pose_euler[0], pose_euler[1], pose_euler[2]
        )
        pose_cmd = PoseStamped()
        pose_cmd.header.stamp = rospy.Time.now()
        pose_cmd.header.frame_id = "hand_palm_link"
        # pose_cmd.pose.position.x = self.prev_pos_trans.x
        # pose_cmd.pose.position.y = self.prev_pos_trans.y
        # pose_cmd.pose.position.z = self.prev_pos_trans.z
        # pose_cmd.pose.orientation.x = self.prev_pos_quat.x
        # pose_cmd.pose.orientation.y = self.prev_pos_quat.y
        # pose_cmd.pose.orientation.z = self.prev_pos_quat.z
        # pose_cmd.pose.orientation.w = self.prev_pos_quat.w
        pose_cmd.pose.position.x = self.prev_pos_trans.x + pose_trans[0]
        pose_cmd.pose.position.y = self.prev_pos_trans.y + pose_trans[1]
        pose_cmd.pose.position.z = self.prev_pos_trans.z + pose_trans[2]
        pose_cmd.pose.orientation.x = self.prev_pos_quat.x + pose_quaternion[0]
        pose_cmd.pose.orientation.y = self.prev_pos_quat.y + pose_quaternion[1]
        pose_cmd.pose.orientation.z = self.prev_pos_quat.z + pose_quaternion[2]
        pose_cmd.pose.orientation.w = self.prev_pos_quat.w + pose_quaternion[3]
        if to_ros:
            self.pose_pub_2.publish(pose_cmd)

        trigger_cmd_1 = Float32()
        trigger_cmd_1.data = 0.0
        if to_ros:
            self.trigger_pub_1.publish(trigger_cmd_1)

        pose_action = pose_action.to("cpu").detach().numpy().astype(np.float64).copy()
        trigger_cmd_2 = Float32()
        trigger_cmd_2.data = 1.0
        if to_ros:
            self.trigger_pub_2.publish(trigger_cmd_2)

        self.prev_pos_trans = pose_cmd.pose.position
        self.prev_pos_quat = pose_cmd.pose.orientation
        # print(step, self.prev_pos_trans)
        print(step, pose_trans)


    def get_action(self, step):
        """get action from dataset

        Returns:
            actions: torch action
        """
        # (head_images, hand_images, joint_states), (
        #     base_cmd,
        #     arm_trans,
        #     arm_angle,
        #     arm_action,
        # ) = next(iter(train_dataloader))
        # print(self.data["base_cmd"])
        base_cmd = self.data["base_cmd"][step]
        arm_trans = self.data["arm_pose"][step+1][:3] - self.data["arm_pose"][step][:3]
        arm_angle = self.data["arm_pose"][step+1][3:] - self.data["arm_pose"][step][3:]
        arm_action = self.data["arm_action"][step]
        # print(step, arm_trans)
        return base_cmd, arm_trans, arm_angle, arm_action

    def head_rgb_callback(self, msg):
        self.head_rgb_msg = msg

    def hand_rgb_callback(self, msg):
        self.hand_rgb_msg = msg

    def joint_states_callback(self, msg):
        self.joint_states_msg = msg


if __name__ == "__main__":
    rospy.init_node("model_agent")


    config = rospy.get_param(
        "/data/config", "/root/catkin_ws/src/dl_ros_for_mm/configs/config.json"
    )
    device = rospy.get_param("/device", "cuda")

    device = select_device(device)
    import pickle

    with open(
        "/root/catkin_ws/src/dl_ros_for_mm/dataset/run_and_grasp/run_and_grasp.pkl", "rb"
        # "/root/catkin_ws/src/dl_ros_for_mm/dataset/arm/arm.pkl", "rb"
    ) as f:
        loaded_data = pickle.load(f)
    # print(loaded_data[0]["arm_action"])
    # train_dataset = MyDataset(data=loaded_data[0], noise=0.005)
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=1, shuffle=True, num_workers=1
    # )

    agent = DemoAgent(device=device, data=loaded_data[0])
