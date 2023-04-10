import argparse

import torch

from model.bc import BCAgent
from model.env import Env
from model.model_agent import ModelAgent
from utils.device import select_device


def inference(args):
    device = select_device(args.device)
    model = BCAgent()
    model.load_state_dict(torch.load("best_weight.pth"))
    model.to(device)
    env = Env(config=args.config)
    agent = ModelAgent(model=model, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()

    inference(args)
