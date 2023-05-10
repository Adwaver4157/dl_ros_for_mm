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

from utils.utils import convert_Image, convert_JointStates, torch_fix_seed

import argparse

import numpy as np
import torch
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Quaternion
from std_msgs.msg import Float32

from model.base_agent import BaseAgent
from model.bc import BCAgent, BCAgentNoCNN
from model.env import Env
from utils.device import select_device
from utils.dataset import MyDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2
import pickle


def visualize_movie(file_path="dataset/run_and_grasp/run_and_grasp.pkl"):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    # data_type = "head_images"
    data_type = "hand_images"
    images = loaded_data[0][data_type]
    print(f"images steps: {len(images)}")

    fig, ax = plt.subplots()

    def update(image):
        numpy_image = (image.to("cpu").detach().numpy()).astype(np.uint8).copy()
        image = torch.Tensor(cv2.resize(numpy_image, (224, 224)))
        print(image.shape)
        if data_type == "head_images":
            # headがなぜかbgrがデフォルト
            image = image.permute(2, 1, 0).permute(2, 1, 0)
            # bgr = numpy_image.transpose(2, 1, 0)
        else:
            image = image.permute(2, 1, 0).permute(2, 1, 0).flip(2)
        bgr = image.to("cpu").detach().numpy().astype(np.uint8).copy()
        rgb = bgr[..., ::-1].copy()
        plt.clf()
        plt.imshow(rgb)

    anim = animation.FuncAnimation(fig, update, frames=images, interval=100)

    plt.show()


class ModelAgent(BaseAgent):
    def __init__(
        self,
        model,
        env,
        config="/root/catkin_ws/src/dl_ros_for_mm/configs/config.json",
        device="cuda",
        # steps=25,
        steps=150,
        hz=10,
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
        self.head_rgb_msg = None
        self.hand_rgb_msg = None
        self.joint_states_msg = None
        self.index = 0
        self.steps = steps
        rate = rospy.Rate(hz)

        self.image_lst = []

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
            "/hsrb/robot_state/joint_states",
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
        self.model.train()
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
        self.prev_pos_trans = Vector3(0, 0, 0)
        self.prev_pos_quat = Quaternion(0, 0, 0, 1)

        try:
            while not rospy.is_shutdown():
                if self.index < self.steps:
                    self.step()
                else:
                    break
                rate.sleep()
            trigger_cmd_2 = Float32()
            trigger_cmd_2.data = 0.0
            self.trigger_pub_2.publish(trigger_cmd_2)

        except KeyboardInterrupt:
            print("Shutting down")
            base_cmd = Twist()
            base_cmd.linear.x = 0.0
            base_cmd.angular.z = 0.0
            self.base_pub.publish(base_cmd)
        rospy.spin()

    def step(self):
        obs = self.get_obs()
        if obs is not None:
            head_image = obs["head_image"].to(self.device)
            hand_image = obs["hand_image"].to(self.device)
            joint_state = obs["joint_state"].to(self.device)
            base_vel, pose_trans, pose_angle, pose_action = self.model(
                head_image, hand_image, joint_state
            )
            # pose_trans, pose_angle = self.model(joint_state)

            # base_vel = base_vel.to("cpu").detach().numpy().astype(np.float64).copy()
            # print(base_vel)
            # base_cmd = Twist()
            # base_cmd.linear.x = base_vel[0][0]
            # base_cmd.angular.z = base_vel[0][1]

            # to_ros = False
            to_ros = True
            # if to_ros:
            # self.base_pub.publish(base_cmd)

            pose_trans = pose_trans.to("cpu").detach().numpy().astype(np.float64).copy()
            pose_euler = pose_angle.to("cpu").detach().numpy().astype(np.float64).copy()
            pose_quaternion = tf.transformations.quaternion_from_euler(
                pose_euler[0][0], pose_euler[0][1], pose_euler[0][2]
            )
            pose_cmd = PoseStamped()
            pose_cmd.header.stamp = rospy.Time.now()
            pose_cmd.header.frame_id = "hand_palm_link"
            pose_cmd.pose.position.x = self.prev_pos_trans.x + pose_trans[0][0]
            pose_cmd.pose.position.y = self.prev_pos_trans.y + pose_trans[0][1]
            pose_cmd.pose.position.z = self.prev_pos_trans.z + pose_trans[0][2]
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

            # pose_action = (
            #     pose_action.to("cpu").detach().numpy().astype(np.float64).copy()
            # )
            trigger_cmd_2 = Float32()
            # # print(pose_action[0][0])
            # trigger_cmd_2.data = pose_action[0][0]
            trigger_cmd_2.data = 1.0
            if to_ros:
                self.trigger_pub_2.publish(trigger_cmd_2)

            self.prev_pos_trans = pose_cmd.pose.position
            self.prev_pos_quat = pose_cmd.pose.orientation
            # print(self.prev_pos_trans)
            print(self.index, pose_trans)
            self.index += 1

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
            # print(self.head_rgb_msg)
            hand_rgb = convert_Image(
                self.hand_rgb_msg, self.config["height"], self.config["width"]
            )
            # print(self.hand_rgb_msg)

            # plt.imshow(head_rgb[0])
            # plt.show()
            # # print(head_rgb[0])
            # plt.imshow(hand_rgb[0])
            # plt.show()
            # print(hand_rgb[0])
            joint_states = convert_JointStates(self.joint_states_msg)
            torch_data["head_image"] = (
                torch.tensor(head_rgb[0], dtype=torch.float32).flip(2).permute(2, 1, 0)
            ).unsqueeze(0)
            # head_rgb = torch_data["head_image"].squeeze(0).permute(2,1,0).to("cpu").detach().numpy().astype(np.uint8).copy()
            # plt.imshow(head_rgb)
            # plt.show()
            # print(torch_data["head_image"])
            # print(head_rgb)
            
            torch_data["hand_image"] = (
                torch.tensor(hand_rgb[0], dtype=torch.float32)
                .permute(2, 1, 0)  
            ).unsqueeze(0)
            # hand_rgb = torch_data["hand_image"].squeeze(0).permute(2,1,0).to("cpu").detach().numpy().astype(np.uint8).copy()
            # plt.imshow(hand_rgb)
            # plt.show()
            # print(torch_data["hand_image"])
            # print(hand_rgb)

            if self.transform:
                torch_data["head_image"] = self.transform(torch_data["head_image"] / 255)
                torch_data["hand_image"] = self.transform(torch_data["hand_image"] / 255)
            # print(torch_data["head_image"])
            # print(torch_data["hand_image"])

            # visualize = True
            visualize = False
            if visualize:
                self.update(torch_data["head_image"])
                plt.show()
                self.update(torch_data["hand_image"])
                plt.show()
                # self.image_lst.append(torch_data["hand_image"])
                # if len(self.image_lst) % 50:
                #     fig, ax = plt.subplots()
                #     anim = animation.FuncAnimation(
                #         fig, self.update, frames=self.image_lst, interval=100
                #     )

                #     plt.show()

            torch_data["joint_state"] = torch.tensor(joint_states, dtype=torch.float32)

            return torch_data
        else:
            return None

    def update(self, image):
        data_type = "hand_image"
        print(image.shape)
        numpy_image = (
            (image.squeeze(0).permute(2, 1, 0).to("cpu").detach().numpy())
            .astype(np.uint8)
            .copy()
        )
        plt.clf()
        plt.imshow(numpy_image)

    def head_rgb_callback(self, msg):
        self.head_rgb_msg = msg

    def hand_rgb_callback(self, msg):
        self.hand_rgb_msg = msg

    def joint_states_callback(self, msg):
        self.joint_states_msg = msg


if __name__ == "__main__":
    torch_fix_seed()
    rospy.init_node("model_agent")

    config = rospy.get_param(
        "/data/config", "/root/catkin_ws/src/dl_ros_for_mm/configs/config.json"
    )
    device = rospy.get_param("/device", "cuda")

    device = select_device("cpu")
    model = BCAgent()
    # model = BCAgentNoCNN()
    model.load_state_dict(
        torch.load(
            # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_arm/best_model.pth",
            # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_run_and_grasp/model_35.pth",
            "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_run_and_grasp_w_cnn_half/best_model.pth",
            # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_run_and_grasp_w_cnn/best_model.pth",
            map_location="cpu",
        )
    )
    model.to(device)
    env = Env(config=config)
    agent = ModelAgent(model=model, env=env, device=device, steps=100)
