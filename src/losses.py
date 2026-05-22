"""Losses for super-resolution.

- L1 pixel loss (MAE).
- VGG19 perceptual loss on features {relu1_2, relu2_2, relu3_3}.
- CombinedLoss = w_l1 * L1 + w_p * VGGPerceptualLoss.

Inputs/targets are expected in pixel range [0, 1]. The VGG branch
internally rescales to the ImageNet mean/std the network was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG19_Weights, vgg19

# Layer indices in torchvision's `vgg19().features` for the ReLUs that
# follow each named conv block. (relu1_2=3, relu2_2=8, relu3_3=15)
VGG_FEATURE_LAYERS = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss = mean L1 distance between selected VGG19 features."""

    def __init__(self, layer_names: tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3")) -> None:
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        full = vgg19(weights=weights).features.eval()
        for p in full.parameters():
            p.requires_grad_(False)
        max_idx = max(VGG_FEATURE_LAYERS[n] for n in layer_names)
        self.features = full[: max_idx + 1]
        self.targets = [VGG_FEATURE_LAYERS[n] for n in layer_names]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "VGGPerceptualLoss":
        # Force eval mode regardless of parent .train() calls.
        return super().train(False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.targets:
                feats.append(x)
                if i == self.targets[-1]:
                    break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)
        with torch.no_grad():
            target_feats = self._extract(target_n)
        pred_feats = self._extract(pred_n)
        loss = sum(F.l1_loss(p, t) for p, t in zip(pred_feats, target_feats))
        return loss / len(pred_feats)


@dataclass
class LossWeights:
    l1: float = 1.0
    perceptual: float = 0.1


class CombinedLoss(nn.Module):
    """w_l1 * L1 + w_p * VGGPerceptual."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.l1 = nn.L1Loss()
        self.perceptual = VGGPerceptualLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = self.l1(pred, target)
        perc = self.perceptual(pred, target)
        total = self.weights.l1 * l1 + self.weights.perceptual * perc
        return total, {"l1": l1.detach(), "perceptual": perc.detach(), "total": total.detach()}
