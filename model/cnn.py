#!/usr/bin/python3

import pickle
import argparse

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, embed_dim=64, debug=False):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(512, embed_dim)
        self.debug = debug

    def forward(self, x):
        out = self.resnet18(x)
        if self.debug:
            print(f"out: {out}")
        return self.resnet18(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model = ResNet18(debug=args.debug)
    # print(model)
    with open("dataset/arm/arm.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    # data_type = "head_images"
    data_type = "hand_images"

    test_image = loaded_data[0][data_type][10]
    if data_type == "head_images":
        test_image = test_image.flip(2).permute(2, 1, 0)
    else:
        test_image = test_image.permute(2, 1, 0)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(test_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    model.train()
    with torch.no_grad():
        output = model(input_batch)
    print(f"train mode: {output}")

    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    print(f"eval mode: {output}")
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)
