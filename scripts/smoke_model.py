"""Smoke test for UNetSR: forward pass shape + param count."""
from __future__ import annotations

import torch

from src.model import UNetSR, count_parameters


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for base in (32, 48, 64):
        model = UNetSR(base_channels=base).to(device)
        total, trainable = count_parameters(model)
        x = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 3, 128, 128), f"unexpected output shape {y.shape}"
        assert 0.0 <= y.min().item() and y.max().item() <= 1.0, "sigmoid out of range"
        print(
            f"base={base:>3d}  params total={total/1e6:6.2f}M trainable={trainable/1e6:6.2f}M  "
            f"in={tuple(x.shape)} out={tuple(y.shape)} range=[{y.min():.3f}, {y.max():.3f}]"
        )

    # Default config summary
    model = UNetSR().to(device)
    print("\n[architecture]\n", model)


if __name__ == "__main__":
    main()
