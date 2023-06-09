import argparse
import copy
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from model.bc import BCAgent, BCAgentNoCNN
from utils.dataset import MyDataset, split_dataset
from utils.device import select_device
from utils.utils import torch_fix_seed


def train(args):
    if os.path.exists(f"weight/{args.name}"):
        print("Warning: weight folder already exists")
    else:
        os.makedirs(f"weight/{args.name}", exist_ok=True)

    with open(f"dataset/{args.dataset_name}/{args.dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),  # type: ignore
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data, test_data = split_dataset([loaded_data[0]], args.split_ratio)
    # train_dataset = MyDataset(data=[train_data[0]], transform=preprocess)
    train_dataset = MyDataset(
        data=train_data, transform=preprocess, noise=args.action_noise
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_dataset = MyDataset(data=test_data, transform=preprocess)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    dataloaders = {"train": train_loader, "val": test_loader}

    device = select_device(args.device)
    if args.model_name == "cnn":
        model = BCAgent().to(device)
    elif args.model_name == "wo_cnn":
        model = BCAgentNoCNN().to(device)
        model.load_state_dict(
            torch.load(
                # "/root/catkin_ws/src/dl_ros_for_mm/arm_best.pth",
                "/root/catkin_ws/src/dl_ros_for_mm/arm_best_wo_cnn.pth",
                # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_arm/best_model.pth",
                map_location="cuda",
            )
        )
        model = model.to(device)
    # print(model)
    huber_loss = nn.HuberLoss()
    # log_loss = nn.NLLLoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=0)

    since = time.time()

    val_loss_history = []

    best_loss = 10
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch, args.epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for batch_idx, (inputs, target) in enumerate(dataloaders[phase]):
                inputs = list(map(lambda x: x.to(device), inputs))
                target = list(map(lambda x: x.to(device), target))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # if 5 < batch_idx < 15:
                #     print(batch_idx)
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    # print(inputs[2].shape)
                    if args.model_name == "cnn":
                        base_cmd, arm_trans, arm_angle, arm_action = model(*inputs)
                        
                        print(f"base vel: {base_cmd}")
                        loss_base = huber_loss(base_cmd, target[0])
                        loss_arm_trans = huber_loss(arm_trans, target[1])
                        loss_arm_angle = huber_loss(arm_angle, target[2])
                        loss_arm_action = ce_loss(arm_action, target[3].squeeze(1).long())
                        loss = (
                            loss_base
                            + loss_arm_trans
                            + loss_arm_angle
                            + 10 * loss_arm_action
                        )
                    elif args.model_name == "wo_cnn":
                        arm_trans, arm_angle = model(inputs[-1])
                        print(f"trans ans: {target[1]}, pred: {arm_trans}")
                        print()
                        print(f"angle ans: {target[2]}, pred: {arm_angle}")
                        loss_arm_trans = huber_loss(arm_trans, target[1])
                        loss_arm_angle = huber_loss(arm_angle, target[2])
                        loss = (
                            loss_arm_trans
                            + loss_arm_angle
                        )
                    
                    

                    

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs[0].size(0)

            epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore

            print(f"{phase} Loss: {epoch_loss:.4f}")

            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"weight/{args.name}/model_{epoch}.pth")

            # deep copy the model
            print(epoch_loss)
            print(best_loss)
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"weight/{args.name}/best_model.pth")

            val_loss_history.append(epoch_loss)
        scheduler.step()

    print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best loss: {:4f}".format(best_loss))
    torch.save(best_model_wts, f"weight/{args.name}/best_model.pth")
    print(f"save in weight/{args.name}/best_model.pth")

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    print("### Train model test ###")
    model.eval()
    arm_trans, arm_angle = model(inputs[-1])
    print(f"trans ans: {target[1]}, pred: {arm_trans}")
    print()
    print(f"angle ans: {target[2]}, pred: {arm_angle}")
    weight = model.state_dict()
    
    model_eval = BCAgentNoCNN().to(device)
    model_eval.load_state_dict(weight)
    # model_eval.load_state_dict(
    #     torch.load(
    #         # "/root/catkin_ws/src/dl_ros_for_mm/arm_best.pth",
    #         # "/root/catkin_ws/src/dl_ros_for_mm/arm_best_wo_cnn.pth",
    #         # "/root/catkin_ws/src/dl_ros_for_mm/weight/BC_arm/best_model.pth",
    #         f"weight/{args.name}/best_model.pth",
    #         # map_location="cuda",
    #     )
    # )
    model_eval = model_eval.to(device)
    model_eval.eval()
    arm_trans, arm_angle = model_eval(inputs[-1])
    
    print("### Eval model test ###")
    print(f"trans ans: {target[1]}, pred: {arm_trans}")
    print()
    print(f"angle ans: {target[2]}, pred: {arm_angle}")
    
    print(f"weight: \n {weight.keys()}")
    print(f"weight: \n {len(weight)}")
    torch.save(weight, "weight/test.pth")
    # save_weight = torch.load("weight/test.pth")
    save_weight = torch.load("weight/BC_arm/best_model.pth")
    print(f"save weight: \n {save_weight.keys()}")
    print(f"save weight: \n {len(save_weight)}")
    
    # for k in weight.keys():
    #     # print(weight[k])
    #     # print(save_weight[k])
    #     weight_i = weight[k].to("cpu").detach().numpy().astype(np.float64).tolist()
    #     save_weight_i = save_weight[k].to("cpu").detach().numpy().astype(np.float64).tolist()
    #     print(np.array(weight_i) - np.array(save_weight_i))
    #     if weight_i == save_weight_i:
    #         print(f"{k} Same")
    #     else:
    #         print(np.array(weight_i) - np.array(save_weight_i))
    #         print(f"{k} not same")           
            
    


if __name__ == "__main__":
    torch_fix_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="BC_arm")
    parser.add_argument("--exp_name", type=str, default="wo_cnn")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=5e-3)
    # parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--action_noise", type=float, default=0.000)
    parser.add_argument("--dataset_name", type=str, default="arm")
    parser.add_argument("--model_name", type=str, default="cnn", choices=["cnn", "wo_cnn"])
    args = parser.parse_args()

    train(args)
