"""U-Net super-resolution model v3.

Changes vs. v2 (`src.model_v2.UNetSRv2`):
  1. **RCAB blocks** (Zhang et al., RCAN 2018) replace plain DoubleConv. Each
     block is `[Conv3x3 -> ReLU -> Conv3x3 -> ChannelAttention] + skip`,
     which (a) introduces residual learning *inside* every stage so deeper
     stacks remain trainable and (b) lets the network reweight channels by
     learned global importance — extremely effective for SR texture.
  2. **ICNR initialization** for PixelShuffle convs (Aitken et al. 2017).
     Standard init produces a checkerboard pattern through sub-pixel
     convolution; ICNR replicates a smaller init across the r² subpixel
     groups so the initial upsample is artifact-free. Real visible gain on
     fine textures.
  3. Same residual-over-bicubic skip + zero-init head as v2, so v3 also
     starts at the bicubic baseline (PSNR ~28-29 dB at epoch 0).

Everything else (loss, optimizer, dataset, augmentations) stays unchanged
from v2 so the gain is attributable to architecture only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ─────────────────────────────────────────────────────────────────────────────
# Channel attention + RCAB
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention (RCAN variant)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class RCAB(nn.Module):
    """Residual Channel Attention Block: Conv-ReLU-Conv-CA + identity skip."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ChannelAttention(channels, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCAStage(nn.Module):
    """Channel-projecting stage: 3x3 proj to `out_ch` + RCAB. Drop-in
    replacement for the v1/v2 DoubleConv (same input/output signature)."""

    def __init__(self, in_ch: int, out_ch: int, reduction: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.rcab = RCAB(out_ch, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rcab(self.act(self.proj(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Encoder/decoder building blocks
# ─────────────────────────────────────────────────────────────────────────────
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.stage = RCAStage(in_ch, out_ch, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, reduction: int = 16) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = nn.Conv2d(in_ch, skip_ch, kernel_size=1)
        self.stage = RCAStage(skip_ch * 2, out_ch, reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(self.up(x))
        return self.stage(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# ICNR-initialized PixelShuffle block
# ─────────────────────────────────────────────────────────────────────────────
def icnr_init_(tensor: torch.Tensor, scale: int = 2,
               init: callable = nn.init.kaiming_normal_) -> None:
    """ICNR init (Aitken et al.): produces an initial sub-pixel conv that
    behaves like a nearest-neighbor upsample, removing checkerboard at t=0.
    Modifies `tensor` in place. Expects shape (out_ch * scale**2, in_ch, k, k).
    """
    out_ch, in_ch, k1, k2 = tensor.shape
    sub_ch = out_ch // (scale ** 2)
    sub = torch.empty(sub_ch, in_ch, k1, k2)
    init(sub)
    sub = sub.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        tensor.copy_(sub)


class PixelShuffleBlock(nn.Module):
    """Conv -> PixelShuffle(r=2) -> ReLU, with ICNR init on the conv kernel."""

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale ** 2), kernel_size=3, padding=1)
        icnr_init_(self.conv.weight, scale=scale)
        nn.init.zeros_(self.conv.bias)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# UNetSRv3
# ─────────────────────────────────────────────────────────────────────────────
class UNetSRv3(nn.Module):
    """U-Net SR network with RCAB stages, ICNR PixelShuffle, bicubic residual.

    Output = clamp(head(features) + bicubic_upsample(input), 0, 1).
    Head is zero-init so output == bicubic at epoch 0.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 48,
        scale: int = 4,
        ca_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.scale = scale
        b = base_channels
        # Encoder
        self.inc = RCAStage(in_channels, b, ca_reduction)
        self.down1 = Down(b, b * 2, ca_reduction)
        self.down2 = Down(b * 2, b * 4, ca_reduction)
        self.down3 = Down(b * 4, b * 8, ca_reduction)
        # Bottleneck
        self.bottleneck = RCAStage(b * 8, b * 8, ca_reduction)
        # Decoder
        self.up1 = Up(in_ch=b * 8, skip_ch=b * 4, out_ch=b * 4, reduction=ca_reduction)
        self.up2 = Up(in_ch=b * 4, skip_ch=b * 2, out_ch=b * 2, reduction=ca_reduction)
        self.up3 = Up(in_ch=b * 2, skip_ch=b,     out_ch=b,     reduction=ca_reduction)
        # x4 upscale (two x2 stages with ICNR init)
        self.ps1 = PixelShuffleBlock(b, scale=2)
        self.ps2 = PixelShuffleBlock(b, scale=2)
        # Head: 1x1 conv producing the residual. Zero-init so f(x)=bicubic(x) at t=0.
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        y = self.up1(x5, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.ps1(y)
        y = self.ps2(y)
        residual = self.head(y)
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        return (base + residual).clamp(0.0, 1.0)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
