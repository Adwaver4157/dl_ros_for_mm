import pickle

import cv2
import numpy as np
import torch
from torchvision import transforms

from matplotlib import pyplot as plt
import matplotlib.animation as animation



def show_image(image, data_type="head_images"):
    numpy_image = (
        (image.to("cpu").detach().numpy().transpose(2, 1, 0)).astype(np.uint8).copy()
    )
    print(numpy_image.shape)

    # transform_tensor = transforms.PILToTensor()
    # transform = transforms.ToPILImage()
    # bgr = transform(torch.Tensor(numpy_image))
    # rgb = (bgr.to("cpu").detach().numpy().transpose(0, 1, 2)).astype(np.uint8).copy()
    # rgb = (image.to("cpu").detach().numpy().transpose(0, 1, 2)).astype(np.uint8).copy()
    # print(rgb.shape)
    # bgr = cv2.cvtColor(numpy_image.transpose(2, 1, 0), cv2.COLOR_RGB2BGR)
    numpy_image = (image.to("cpu").detach().numpy()).astype(np.uint8).copy()
    image = torch.Tensor(cv2.resize(numpy_image, (224, 224)))
    print(image.shape)
    if data_type == "head_images":
        # headがなぜかbgrがデフォルト
        image = image.permute(2, 1, 0).permute(2, 1, 0)
        # bgr = numpy_image.transpose(2, 1, 0)
    else:
        image = image.permute(2, 1, 0).permute(2, 1, 0).flip(2)
        # bgr = numpy_image.transpose(2, 1, 0)[..., ::-1]
    bgr = image.to("cpu").detach().numpy().astype(np.uint8).copy()

    # print(bgr.size)
    # print(transform_tensor(bgr).shape)
    # bgr.show()
    # bgr.save("test_1.jpg")
    cv2.imwrite("test_2.jpg", bgr)


def visualize_image(file_path="dataset/sim_move/sim_move.pkl"):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    data_type = "head_images"
    # data_type = "hand_images"

    test_image = loaded_data[0][data_type][38]
    print(len(loaded_data[0][data_type]))
    print(test_image.shape)
    show_image(test_image, data_type=data_type)

def visualize_movie(file_path="dataset/sim_move/sim_move.pkl"):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    data_type = "head_images"
    # data_type = "hand_images"
    images = loaded_data[0][data_type]
    print(f'images steps: {len(images)}')

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
        rgb = bgr[...,::-1].copy()
        plt.clf()
        plt.imshow(rgb)

    anim = animation.FuncAnimation(fig, update, frames=images, interval=100)

    plt.show()



if __name__ == "__main__":
    # visualize_image()
    visualize_movie()
