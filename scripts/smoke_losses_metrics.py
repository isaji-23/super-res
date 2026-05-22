"""Sanity check for losses and metrics.

Cases:
  1. Identical images -> L1 ~ 0, perceptual ~ 0, PSNR ~ inf, SSIM ~ 1, LPIPS ~ 0.
  2. Different images (random vs real) -> losses high, PSNR low, SSIM low, LPIPS high.
"""
from __future__ import annotations

import torch

from src.data import DataConfig, DIV2KDataset
from src.losses import CombinedLoss
from src.metrics import lpips_metric, psnr, ssim


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ds = DIV2KDataset(DataConfig(), "val")  # type: ignore[arg-type]
    pairs = [ds[i] for i in range(4)]
    hr = torch.stack([p[1] for p in pairs]).to(device)

    print("\n[case 1] identical images")
    pred = hr.clone()
    loss_fn = CombinedLoss().to(device)
    total, parts = loss_fn(pred, hr)
    print(
        f"  L1={parts['l1'].item():.4e}  perceptual={parts['perceptual'].item():.4e}  "
        f"total={parts['total'].item():.4e}"
    )
    print(
        f"  PSNR={psnr(pred, hr):.2f} dB  SSIM={ssim(pred, hr):.4f}  "
        f"LPIPS={lpips_metric(pred, hr):.4e}"
    )

    print("\n[case 2] random noise vs HR")
    g = torch.Generator(device=device).manual_seed(0)
    rand = torch.rand(hr.shape, device=device, generator=g)
    total, parts = loss_fn(rand, hr)
    print(
        f"  L1={parts['l1'].item():.4f}  perceptual={parts['perceptual'].item():.4f}  "
        f"total={parts['total'].item():.4f}"
    )
    print(
        f"  PSNR={psnr(rand, hr):.2f} dB  SSIM={ssim(rand, hr):.4f}  "
        f"LPIPS={lpips_metric(rand, hr):.4f}"
    )

    print("\n[case 3] heavy bicubic blur (LR-upscaled vs HR)")
    lr = torch.stack([p[0] for p in pairs]).to(device)
    bicubic = torch.nn.functional.interpolate(
        lr, scale_factor=4, mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 1)
    total, parts = loss_fn(bicubic, hr)
    print(
        f"  L1={parts['l1'].item():.4f}  perceptual={parts['perceptual'].item():.4f}  "
        f"total={parts['total'].item():.4f}"
    )
    print(
        f"  PSNR={psnr(bicubic, hr):.2f} dB  SSIM={ssim(bicubic, hr):.4f}  "
        f"LPIPS={lpips_metric(bicubic, hr):.4f}"
    )


if __name__ == "__main__":
    main()
