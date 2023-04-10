import argparse
import json
import os

import torch
from torchvision import transforms
import rospy
from sensor_msgs.msg import Image, JointState

from utils.utils import convert_Image, convert_JointStates

IMAGE_TIMEOUT = 3.0


class Env:
    def __init__(self, config="configs/config.json", transform=None):
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),  # type: ignore
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
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
        if os.path.exists(config):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError("cannot find config file")

    def get_obs(self):
        """get ros image and joint states and convert to torch tensor

        Returns:
            Dict of torch tensors: observations of head image, hand image, joint states
        """
        torch_data = {}
        self.head_rgb_msg = None
        self.hand_rgb_msg = None
        self.joint_states_msg = None

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
                torch.tensor(head_rgb, dtype=torch.float32).flip(2).permute(2, 1, 0)
            )
            torch_data["hand_image"] = torch.tensor(
                hand_rgb, dtype=torch.float32
            ).permute(2, 1, 0)

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
    rospy.init_node("env", anonymous=True)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="configs/config.json")
    args = argparser.parse_args()

    env = Env(config=args.config)
