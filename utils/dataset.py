import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def resize_torch(image, size=(224, 224)):
    numpy_image = (image.to("cpu").detach().numpy()).astype(np.uint8).copy()
    image = torch.Tensor(cv2.resize(numpy_image, size))
    return image


def split_dataset(list_data, ratio=0.8):
    if int(len(list_data) * ratio) == 0:
        split_idx = 1
    elif int(len(list_data) * ratio) == len(list_data):
        split_idx = -1
    else:
        split_idx = int(len(list_data) * ratio)

    train_data = list_data[:split_idx]
    test_data = list_data[split_idx:]
    return train_data, test_data


class MyDataset(Dataset):
    def __init__(self, data, transform=None, noise=0.005):
        super().__init__()
        self.data = self._preprocess(data)
        self.len = len(self.data["head_images"])
        self.transform = transform
        self.noise = noise

    def _preprocess(self, data):
        if type(data) is not list:
            data = [data]
        concat_data = data[0]
        keys = data[0].keys()
        for d in data[1:]:
            for key in keys:
                concat_data[key] = torch.cat((concat_data[key], d[key]), dim=0)
        return concat_data

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        head_image = self.data["head_images"][index]
        hand_image = self.data["hand_images"][index]
        joint_states = self.data["joint_states"][index]
        base_cmd = self.data["base_cmd"][index]
        arm_trans = self.data["arm_pose"][index][:3]
        arm_angle = self.data["arm_pose"][index][3:]
        arm_action = self.data["arm_action"][index].unsqueeze(0)
        # print(f"{index}: {arm_action}")
        # arm_action = torch.where(
        #     self.data["arm_action"][index] < 0.5, 0.0, 1.0
        # ).unsqueeze(0)

        # arm_pose = self.data["arm_pose"][index]
        # noise = torch.normal(0.0, self.noise, size=(arm_pose.size(0),))
        # arm_pose += noise
        noise = torch.normal(0.0, self.noise, size=(arm_trans.size(0),))
        arm_trans += noise
        noise = torch.normal(0.0, self.noise, size=(arm_angle.size(0),))
        arm_angle += noise

        # head_image = resize_torch(head_image)
        # hand_image = resize_torch(hand_image)

        head_image = head_image.flip(2).permute(2, 1, 0)
        hand_image = hand_image.permute(2, 1, 0)

        if self.transform:
            head_image = self.transform(head_image)
            hand_image = self.transform(hand_image)

        return (head_image, hand_image, joint_states), (
            base_cmd,
            arm_trans,
            arm_angle,
            arm_action,
        )


if __name__ == "__main__":
    import pickle

    with open("dataset/sim/sim.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    print(loaded_data[0]["arm_action"])
    train_dataset = MyDataset(data=loaded_data[0], noise=0.005)
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1
    )

    (head_images, hand_images, joint_states), (
        base_cmd,
        arm_trans,
        arm_angle,
        arm_action,
    ) = next(iter(train_dataloader))
