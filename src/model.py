"""U-Net super-resolution model for x4 upscaling.

Architecture:
  * Encoder: 4 stages of [Conv-BN-ReLU x2] + MaxPool, channels grow as
    base * (1, 2, 4, 8). Input LR 32x32 -> bottleneck spatial 2x2.
  * Bottleneck: 2 convs at base*8 channels.
  * Decoder: 4 stages, each Upsample(nearest, x2) -> Conv -> concat skip
    -> [Conv-BN-ReLU x2]. Restores spatial to 32x32 with `base` channels.
  * Upscale x4: two sequential blocks of [Conv -> PixelShuffle(r=2) -> ReLU].
  * Head: 1x1 conv to 3 channels + sigmoid (output range [0, 1]).

`base_channels` is parameterized (default 48) so the parameter count stays
inside the 2-10M target on consumer GPUs. The spec text suggested
64/128/256/512 but that pushes the total well past 10M; the budget rule
wins. Set `base_channels=32` for tight VRAM, 64 for more capacity.
"""
from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """[Conv3x3 -> BN -> ReLU] x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample x2 + 1x1 conv channel reduction + concat skip + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # Reduce channels before concat to keep DoubleConv input modest.
        self.reduce = nn.Conv2d(in_ch, skip_ch, kernel_size=1)
        self.conv = DoubleConv(skip_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(self.up(x))
        return self.conv(torch.cat([x, skip], dim=1))


class PixelShuffleBlock(nn.Module):
    """Conv -> PixelShuffle(r=2) -> ReLU. Spatial x2, channels preserved."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class UNetSR(nn.Module):
    """U-Net super-resolution network with PixelShuffle x4 head."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 48,
    ) -> None:
        super().__init__()
        b = base_channels
        # Encoder
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        # Bottleneck (extra refinement at deepest level)
        self.bottleneck = DoubleConv(b * 8, b * 8)
        # Decoder
        self.up1 = Up(in_ch=b * 8, skip_ch=b * 4, out_ch=b * 4)
        self.up2 = Up(in_ch=b * 4, skip_ch=b * 2, out_ch=b * 2)
        self.up3 = Up(in_ch=b * 2, skip_ch=b, out_ch=b)
        # x4 upscale + head
        self.ps1 = PixelShuffleBlock(b)
        self.ps2 = PixelShuffleBlock(b)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)        # (B, b, 32, 32)
        x2 = self.down1(x1)     # (B, 2b, 16, 16)
        x3 = self.down2(x2)     # (B, 4b, 8, 8)
        x4 = self.down3(x3)     # (B, 8b, 4, 4)
        x5 = self.bottleneck(x4)
        y = self.up1(x5, x3)    # (B, 4b, 8, 8)
        y = self.up2(y, x2)     # (B, 2b, 16, 16)
        y = self.up3(y, x1)     # (B, b, 32, 32)
        y = self.ps1(y)         # (B, b, 64, 64)
        y = self.ps2(y)         # (B, b, 128, 128)
        return self.out_act(self.head(y))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
