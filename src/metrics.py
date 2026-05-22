"""Image quality metrics for super-resolution evaluation.

Conventions: inputs in pixel range [0, 1], shape (B, 3, H, W), torch
tensors. PSNR/SSIM are reported per-batch averages. LPIPS uses the
VGG-based LPIPS network with inputs rescaled to [-1, 1].
"""
from __future__ import annotations

import math
from typing import Optional

import lpips as _lpips_pkg
import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim

_LPIPS_NET: Optional["_lpips_pkg.LPIPS"] = None


def _to_uint8_np(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor in [0,1] -> (H, W, 3) uint8 numpy."""
    arr = img.detach().clamp(0, 1).mul(255).round().byte()
    return arr.permute(1, 2, 0).cpu().numpy()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Mean PSNR over the batch (per-image then averaged)."""
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
    # Avoid log(0): return inf when MSE == 0 for an image.
    psnrs = torch.where(
        mse > 0,
        10.0 * torch.log10(data_range ** 2 / mse),
        torch.full_like(mse, float("inf")),
    )
    finite = psnrs[torch.isfinite(psnrs)]
    if finite.numel() == 0:
        return float("inf")
    return finite.mean().item()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SSIM over the batch using skimage (channel_axis=-1)."""
    assert pred.shape == target.shape
    vals = []
    for i in range(pred.size(0)):
        p = _to_uint8_np(pred[i])
        t = _to_uint8_np(target[i])
        vals.append(
            _ssim(t, p, channel_axis=-1, data_range=255)
        )
    return float(np.mean(vals))


def _get_lpips(device: torch.device) -> "_lpips_pkg.LPIPS":
    global _LPIPS_NET
    if _LPIPS_NET is None or next(_LPIPS_NET.parameters()).device != device:
        net = _lpips_pkg.LPIPS(net="vgg", verbose=False)
        for p in net.parameters():
            p.requires_grad_(False)
        _LPIPS_NET = net.to(device).eval()
    return _LPIPS_NET


def lpips_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean LPIPS (VGG) over the batch. Lower is better."""
    assert pred.shape == target.shape
    device = pred.device
    net = _get_lpips(device)
    # LPIPS expects inputs in [-1, 1].
    p = pred.mul(2).sub(1)
    t = target.mul(2).sub(1)
    with torch.no_grad():
        d = net(p, t)
    return d.mean().item()
