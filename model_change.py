import torch

# import sys
# from os.path import dirname, abspath

# parent_dir = dirname(dirname(abspath(__file__)))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

from model.bc import BCAgent


def change_model_device(
    model_path="best_model.pth", save_path="best_model_cpu.pth", device="cpu"
):
    model = BCAgent()
    model.load_state_dict(torch.load(model_path))
    torch.save(model.to("cpu").state_dict(), save_path)


if __name__ == "__main__":
    change_model_device()
