"""Sanity-check the DIV2K pipeline.

Saves an 8-pair LR/HR grid to outputs/phase2_batch.png and prints batch
shapes + value ranges for the three splits.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data import DataConfig, DIV2KDataset, get_dataloaders


def _save_grid(lr: torch.Tensor, hr: torch.Tensor, out_path: Path, n: int = 8) -> None:
    n = min(n, lr.size(0))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(lr[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[0, i].set_title(f"LR {tuple(lr[i].shape[-2:])}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(hr[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[1, i].set_title(f"HR {tuple(hr[i].shape[-2:])}", fontsize=8)
        axes[1, i].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    cfg = DataConfig()
    print("[info] split sizes:")
    for split in ("train", "val", "test"):
        ds = DIV2KDataset(cfg, split)  # type: ignore[arg-type]
        print(f"  {split:5s}: {len(ds)} samples (from {len(ds.files)} images)")

    loaders = get_dataloaders(cfg, batch_size=8, num_workers=2)
    for split, loader in loaders.items():
        lr, hr = next(iter(loader))
        print(
            f"[{split}] LR={tuple(lr.shape)} dtype={lr.dtype} "
            f"range=[{lr.min():.3f}, {lr.max():.3f}]  "
            f"HR={tuple(hr.shape)} range=[{hr.min():.3f}, {hr.max():.3f}]"
        )
        if split == "train":
            _save_grid(lr, hr, Path("outputs") / "phase2_batch.png")

    # Determinism check: re-iterate val loader, compare first batch.
    val_loader = loaders["val"]
    lr1, hr1 = next(iter(val_loader))
    lr2, hr2 = next(iter(val_loader))
    same = torch.allclose(lr1, lr2) and torch.allclose(hr1, hr2)
    print(f"[determinism] val first-batch identical across iterations: {same}")

    print("[ok] grid saved to outputs/phase2_batch.png")


if __name__ == "__main__":
    main()
