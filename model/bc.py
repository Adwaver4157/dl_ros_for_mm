import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model.cnn import ResNet18
from model.mlp import MLP
from utils.dataset import MyDataset


class BCAgent(nn.Module):
    def __init__(
        self,
        mlp_input_dim=39,
        mlp_hidden_dims=[50, 50],
        mlp_output_dim=40,
        cnn_embed_dim=64,
        action_embed_dims=[256, 256],
        base_cmd_dims=2,
        arm_trans_dims=3,
        arm_angle_dims=3,
        arm_action_dims=1,
    ):
        super(BCAgent, self).__init__()
        self.mlp = MLP(
            input_dims=mlp_input_dim,
            hidden_dims=mlp_hidden_dims,
            output_dims=mlp_output_dim,
        )
        self.head_cnn = ResNet18(embed_dim=cnn_embed_dim)
        self.hand_cnn = ResNet18(embed_dim=cnn_embed_dim)
        self.base_cmd_output = MLP(
            input_dims=2 * cnn_embed_dim + mlp_output_dim,
            hidden_dims=action_embed_dims,
            output_dims=base_cmd_dims,
        )
        self.arm_trans_output = MLP(
            input_dims=2 * cnn_embed_dim + mlp_output_dim,
            hidden_dims=action_embed_dims,
            output_dims=arm_trans_dims,
        )
        self.arm_angle_output = MLP(
            input_dims=2 * cnn_embed_dim + mlp_output_dim,
            hidden_dims=action_embed_dims,
            output_dims=arm_angle_dims,
        )
        self.arm_action_output = MLP(
            input_dims=2 * cnn_embed_dim + mlp_output_dim,
            hidden_dims=action_embed_dims,
            output_dims=arm_action_dims,
        )

    def forward(self, head_image, hand_image, joint_states):
        head_embed = self.head_cnn(head_image)
        hand_embed = self.hand_cnn(hand_image)
        obs_embed = self.mlp(joint_states)
        x = torch.cat([head_embed, hand_embed, obs_embed], dim=1)
        base_cmd = self.base_cmd_output(x)
        arm_trans = self.arm_trans_output(x)
        arm_angle = self.arm_angle_output(x)
        arm_action = self.arm_action_output(x)
        arm_action = torch.cat([arm_action, 1 - arm_action], dim=1)
        return base_cmd, arm_trans, arm_angle, arm_action


if __name__ == "__main__":
    model = BCAgent()
    # print(model)
    with open("dataset/sim/sim.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),  # type: ignore
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = MyDataset(data=loaded_data, transform=preprocess)
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4
    )

    (head_images, hand_images, joint_states), (
        base_cmd,
        arm_trans,
        arm_angle,
        arm_action,
    ) = next(iter(train_dataloader))

    with torch.no_grad():
        output = model(head_images, hand_images, joint_states)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    # print(output[0])
