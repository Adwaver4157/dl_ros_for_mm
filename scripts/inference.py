#!/usr/bin/python3

import argparse

import torch

import rospy
import sys
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model.bc import BCAgent
from model.env import Env
from model.model_agent import ModelAgent
from utils.device import select_device


def inference(config, device):
    device = select_device(device)
    model = BCAgent()
    model.load_state_dict(torch.load("/root/catkin_ws/src/dl_ros_for_mm/best_model.pth", map_location="cuda:0"))
    model.to(device)
    env = Env(config=config)
    agent = ModelAgent(model=model, env=env)


if __name__ == "__main__":
    config = rospy.get_param('/data/config', "/root/catkin_ws/src/dl_ros_for_mm/configs/config.json")
    device = rospy.get_param('/device', "cuda")

    inference(config, device)
