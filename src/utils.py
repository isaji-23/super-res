"""Shared utilities: seeding and plotting."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_curves(history: dict[str, list[float]], out_path: Path) -> None:
    """Plot train/val loss + val PSNR/SSIM/LPIPS in a 2x2 figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].plot(epochs, history["train_loss"], label="train")
    axes[0, 0].plot(epochs, history["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, history["val_psnr"], color="tab:green")
    axes[0, 1].set_title("Val PSNR (dB)")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, history["val_ssim"], color="tab:orange")
    axes[1, 0].set_title("Val SSIM")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, history["val_lpips"], color="tab:red")
    axes[1, 1].set_title("Val LPIPS (lower = better)")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_history_json(history: dict[str, list[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(history, f, indent=2)
