#!/usr/bin/python3

import pickle

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dims=39, hidden_dims=[50, 50], output_dims=40):
        super().__init__()
        hidden_dims = [input_dims] + hidden_dims
        self.flatten = nn.Flatten()
        layers = []
        for idx in range(len(hidden_dims) - 1):
            layers += [
                nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]),
                nn.ReLU(inplace=True),
            ]
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dims)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    model = MLP(input_dims=39)
    with open("dataset/test_1.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    # data_type = "head_images"
    data_type = "joint_states"

    test_obs = loaded_data[0][data_type][38]
    input_batch = test_obs.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0])
