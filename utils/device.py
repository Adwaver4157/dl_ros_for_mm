import torch


def select_device(device_name="cpu"):
    device = "cpu"
    if device_name == "cpu":
        print("Using CPU.")
    elif device_name == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA.")
        else:
            print("CUDA not available.")
    elif device_name == "mps":
        if not torch.backends.mps.is_available():  # type: ignore[attr-defined]
            if not torch.backends.mps.is_built():  # type: ignore[attr-defined]
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        else:
            print("MPS is available.")
            device = "mps"
    return device


if __name__ == "__main__":
    device = select_device("mps")
    print(device)
