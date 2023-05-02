#!/usr/bin/env python3

import json
import os
import sys
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from torchvision import transforms

import numpy as np
import torch
import rospy

from model.base_agent import BaseAgent
from model.bc import BCAgent, BCAgentNoCNN
from utils.dataset import MyDataset, split_dataset
from model.env import Env
from utils.device import select_device
from torch.utils.data import DataLoader
from utils.utils import torch_fix_seed


def mdoel_inference_test(device="cpu", mode="train"):
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),  # type: ignore
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    import pickle

    with open("dataset/arm/arm.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    train_data, test_data = split_dataset([loaded_data[0]], ratio=0.8)
    # train_dataset = MyDataset(data=[train_data[0]], transform=preprocess)
    train_dataset = MyDataset(data=train_data, transform=preprocess, noise=0)
    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=False, num_workers=0)
    test_dataset = MyDataset(data=test_data, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=0)

    # dataloaders = {"train": train_loader, "val": test_loader}

    device = select_device(device)
    model = BCAgent(debug=False).to(device)
    # model = BCAgentNoCNN(debug=False).to(device)
    model.load_state_dict(
        torch.load(
            # "/root/catkin_ws/src/dl_ros_for_mm/arm_best.pth",
            # "/root/catkin_ws/src/dl_ros_for_mm/arm_best_wo_cnn.pth",
            "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_arm/best_model.pth",
            map_location="cuda",
        )
    )
    model = model.to(device)

    if mode == "train":
        model.train()
        print("start inference in train mode")
    else:
        model.eval()
        print("start inference in eval mode")


    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = list(map(lambda x: x.to(device), inputs))
        target = list(map(lambda x: x.to(device), target))

        # zero the parameter gradients

        # forward
        # with torch.set_grad_enabled(False):
        with torch.no_grad():
            base_cmd, arm_trans, arm_angle, arm_action = model(*inputs)
            print(f"base ans: {target[0]}, pred: {base_cmd}")
            print()
            print(f"trans ans: {target[1]}, pred: {arm_trans}")
            print()
            print(f"angle ans: {target[2]}, pred: {arm_angle}")
            print()
            print(f"action ans: {target[3]}, pred: {arm_action}")
            # arm_trans, arm_angle = model(inputs[-1])
            # print(f"trans ans: {target[1]}, pred: {arm_trans}")
            # print()
            # print(f"angle ans: {target[2]}, pred: {arm_angle}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", default="train", help='set mode, [train] or [eval]', type=str, choices=['train', 'eval'])
    args = parser.parse_args()
    torch_fix_seed()
    device = select_device(args.device)
    mdoel_inference_test(device=device, mode=args.mode)
