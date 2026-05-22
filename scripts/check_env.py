"""Quick GPU / PyTorch sanity check."""
import sys

import torch


def main() -> None:
    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA available:{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device:        {props.name} (cc {props.major}.{props.minor})")
        print(f"VRAM total:    {props.total_memory / 1024**3:.2f} GiB")
        free, total = torch.cuda.mem_get_info(idx)
        print(f"VRAM free:     {free / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB")
    else:
        print("WARNING: no CUDA device detected, training will be CPU-only.")


if __name__ == "__main__":
    main()
