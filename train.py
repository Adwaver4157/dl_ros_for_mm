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

import wandb
from model.bc import BCAgent
from utils.dataset import MyDataset, split_dataset
from utils.device import select_device


def train(args):
    wandb.init(
        project=f"{args.name}",
        name=f"run_{args.epochs}_{args.batch_size}_{args.exp_name}",
    )
    if os.path.exists(f"weight/{args.name}"):
        print("Warning: weight folder already exists")
    else:
        os.makedirs(f"weight/{args.name}", exist_ok=True)

    with open("dataset/sim_move/sim_move.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),  # type: ignore
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data, test_data = split_dataset(loaded_data[:2], args.split_ratio)
    # train_dataset = MyDataset(data=[train_data[0]], transform=preprocess)
    train_dataset = MyDataset(
        data=train_data, transform=preprocess, noise=args.action_noise
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_dataset = MyDataset(data=test_data, transform=preprocess)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    dataloaders = {"train": train_loader, "val": test_loader}

    device = select_device(args.device)
    model = BCAgent().to(device)
    # print(model)
    huber_loss = nn.HuberLoss()
    # log_loss = nn.NLLLoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=0)

    # Magic
    wandb.watch(model, log_freq=100)

    since = time.time()

    val_loss_history = []

    best_loss = 0.0
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
            model.train()
            for batch_idx, (inputs, target) in enumerate(dataloaders[phase]):
                inputs = list(map(lambda x: x.to(device), inputs))
                target = list(map(lambda x: x.to(device), target))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    base_cmd, arm_trans, arm_angle, arm_action = model(*inputs)
                    print(base_cmd)
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

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                if batch_idx % args.log_interval == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "lr": optimizer.param_groups[0]["lr"],
                            f"{phase}_loss": loss,
                            f"{phase}_loss_base": loss_base,
                            f"{phase}_loss_arm_trans": loss_arm_trans,
                            f"{phase}_loss_arm_angle": loss_arm_angle,
                            f"{phase}_loss_arm_action": loss_arm_action,
                        }
                    )

                # statistics
                running_loss += loss.item() * inputs[0].size(0)

            epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore

            print(f"{phase} Loss: {epoch_loss:.4f}")

            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"weight/{args.name}/model_{epoch}.pth")

            # deep copy the model
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

    # # load best model weights
    # model.load_state_dict(best_model_wts)

    # return model, val_acc_history
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="BC_sim_move")
    parser.add_argument("--exp_name", type=str, default="change_weight")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=5e-3)
    # parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--action_noise", type=float, default=0.000)
    args = parser.parse_args()

    train(args)
