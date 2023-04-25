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
from model.bc import BCAgent
from utils.dataset import MyDataset, split_dataset
from model.env import Env
from utils.device import select_device
from torch.utils.data import DataLoader
from utils.utils import torch_fix_seed


def mdoel_inference_test(device="cpu"):
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),  # type: ignore
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    import pickle

    with open("dataset/sim_move/sim_move.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    train_data, test_data = split_dataset([loaded_data[0]], ratio=0.8)
    # train_dataset = MyDataset(data=[train_data[0]], transform=preprocess)
    train_dataset = MyDataset(data=train_data, transform=preprocess, noise=0)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_dataset = MyDataset(data=test_data, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    dataloaders = {"train": train_loader, "val": test_loader}

    device = select_device(device)
    model = BCAgent().to(device)
    model.load_state_dict(
        torch.load(
            # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_sim_move_2/best_model.pth",
            # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_sim_move_2/model_35.pth",
            # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_sim_move_batch_1/model_35.pth",
            "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_sim_move_batch_40/model_35.pth",
            map_location="cpu",
        )
    )
    model = model.to(device)
    phase = "train"
    # model.eval()
    model.train()
    for batch_idx, (inputs, target) in enumerate(dataloaders[phase]):
        inputs = list(map(lambda x: x.to(device), inputs))
        target = list(map(lambda x: x.to(device), target))

        # zero the parameter gradients

        # forward
        # track history if only in train
        # with torch.set_grad_enabled(False):
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            base_cmd, arm_trans, arm_angle, arm_action = model(*inputs)
            print(f"ans: {target[0]}, pred: {base_cmd}")

        # if 5 < batch_idx < 15:
        #     print(batch_idx)
        #     with torch.set_grad_enabled(False):
        #         # Get model outputs and calculate loss
        #         base_cmd, arm_trans, arm_angle, arm_action = model(*inputs)
        #         print(f"ans: {target[0]}, pred: {base_cmd}")


class ModelAgent(BaseAgent):
    def __init__(
        self,
        model,
        env,
        config="/root/catkin_ws/src/dl_ros_for_mm/configs/config.json",
        device="cuda",
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
        self.index = 0
        # print(config)
        if os.path.exists(config):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError("cannot find config file")

        self.model = model
        self.env = env
        import pickle

        with open("dataset/sim_move/sim_move.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        train_dataset = MyDataset(data=loaded_data[0], noise=0)
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        print(train_dataset.len)
        self.model.eval()
        for input, output in train_dataloader:
            input = list(map(lambda x: x.to(device), input))
            with torch.set_grad_enabled(False):
                base_vel, pose_trans, pose_angle, pose_action = self.model(*input)
                print(f"ans: {output[0]}, pred: {base_vel}")


if __name__ == "__main__":
    torch_fix_seed()
    device = select_device("cpu")
    mdoel_inference_test(device=device)

    # rospy.init_node("model_agent")

    # config = rospy.get_param(
    #     "/data/config", "/root/catkin_ws/src/dl_ros_for_mm/configs/config.json"
    # )
    # device = rospy.get_param("/device", "cuda")

    # device = select_device(device)
    # print(device)
    # model = BCAgent()
    # # model.load_state_dict(
    # #     torch.load(
    # #         "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_sim_move/best_model.pth",
    # #         map_location="cuda:0",
    # #     )
    # # )
    # model.load_state_dict(
    #     torch.load(
    #         "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_sim_move_2/best_model.pth",
    #         # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_sim_move_2/model_35.pth",
    #         map_location="cpu",
    #     )
    # )
    # model.to(device)
    # env = Env(config=config)
    # agent = ModelAgent(model=model, env=env, device=device)
