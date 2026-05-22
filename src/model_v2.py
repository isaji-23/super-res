"""U-Net super-resolution model v2 (quick-win iteration over v1).

Changes vs. v1 (`src.model.UNetSR`):
  1. **No BatchNorm** in DoubleConv. BN is known to hurt SR (EDSR/RCAN/
     SwinIR drop it): it normalizes per-batch statistics that the network
     needs to preserve to reconstruct high-frequency texture.
  2. **No final Sigmoid.** Sigmoid saturates gradients near 0 and 1, where
     bright/dark pixels live; replacing it with a clamp keeps gradients
     flowing during training (the clamp is applied to the final output only).
  3. **Residual learning over bicubic.** The convolutional path predicts a
     residual that is added to a bicubic upsample of the LR input. The
     head is zero-initialized so output == bicubic at epoch 0 — PSNR
     starts at the bicubic baseline (~28 dB on DIV2K x4) and the network
     focuses its capacity on high-frequency detail.

Same parameter budget and forward signature as v1, so it is a drop-in
replacement for `src.model.UNetSR`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """[Conv3x3 -> ReLU] x 2. No BatchNorm."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = nn.Conv2d(in_ch, skip_ch, kernel_size=1)
        self.conv = DoubleConv(skip_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(self.up(x))
        return self.conv(torch.cat([x, skip], dim=1))


class PixelShuffleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class UNetSRv2(nn.Module):
    """U-Net SR network with PixelShuffle x4 head + bicubic residual skip."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 48,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.scale = scale
        b = base_channels
        # Encoder
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        # Bottleneck
        self.bottleneck = DoubleConv(b * 8, b * 8)
        # Decoder
        self.up1 = Up(in_ch=b * 8, skip_ch=b * 4, out_ch=b * 4)
        self.up2 = Up(in_ch=b * 4, skip_ch=b * 2, out_ch=b * 2)
        self.up3 = Up(in_ch=b * 2, skip_ch=b, out_ch=b)
        # x4 upscale + head
        self.ps1 = PixelShuffleBlock(b)
        self.ps2 = PixelShuffleBlock(b)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)
        # Zero-init head so output == bicubic baseline at epoch 0.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)         # (B, b,   32, 32)
        x2 = self.down1(x1)      # (B, 2b,  16, 16)
        x3 = self.down2(x2)      # (B, 4b,   8,  8)
        x4 = self.down3(x3)      # (B, 8b,   4,  4)
        x5 = self.bottleneck(x4)
        y = self.up1(x5, x3)     # (B, 4b,   8,  8)
        y = self.up2(y, x2)      # (B, 2b,  16, 16)
        y = self.up3(y, x1)      # (B, b,   32, 32)
        y = self.ps1(y)          # (B, b,   64, 64)
        y = self.ps2(y)          # (B, b,  128, 128)
        residual = self.head(y)
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        return (base + residual).clamp(0.0, 1.0)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
