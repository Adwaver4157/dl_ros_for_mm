#!/usr/bin/python3

import pickle

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, embed_dim=64):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        return self.resnet18(x)


if __name__ == "__main__":
    model = ResNet18()
    print(model)
    with open("dataset/test_1.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    # data_type = "head_images"
    data_type = "hand_images"

    test_image = loaded_data[0][data_type][38]
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

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)
