"""DIV2K dataset and DataLoader factory for x4 super-resolution.

Conventions:
  * Pixel range: [0, 1] (PIL -> float tensor / 255). Matches LPIPS/SSIM
    conventions and pairs naturally with a sigmoid output head.
  * Train: random patch + synchronized augmentations (hflip + 90/180/270).
  * Val/Test: deterministic patches (fixed seed indexed by sample_idx), no
    augmentations.

Optional LR degradation knobs (all opt-in via DataConfig; defaults keep the
clean-bicubic pipeline used by v1/v2/v3):
  * `hr_size` configurable (default 128).
  * `downsample_kind`:
      - "bicubic" → fixed bicubic + antialias (default).
      - "mixed"   → randomly choose per-sample among bicubic/bilinear/area.
        Trains the network to invert a small family of kernels instead of
        memorizing the bicubic one specifically (better generalization to
        real-world LR).
  * `noise_sigma`: if > 0, add gaussian noise to LR (sigma drawn uniformly
    in [0, noise_sigma] per sample, in [0, 1] pixel units).
  * `jpeg_quality_range`: if max < 95, JPEG-compress LR at a random quality
    in this range. Trains tolerance to compression artifacts (web inputs).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

Split = Literal["train", "val", "test"]
DownsampleKind = Literal["bicubic", "mixed"]


@dataclass(frozen=True)
class DataConfig:
    data_root: Path = Path("data")
    hr_size: int = 128
    scale: int = 4
    patches_per_train_image: int = 16  # virtual epoch length multiplier
    val_patches_per_image: int = 1
    test_patches_per_image: int = 1
    val_count: int = 50  # first 50 valid_HR images
    test_count: int = 50  # remaining valid_HR images
    seed: int = 1337
    # Preload all HR images as uint8 tensors once. On Linux, DataLoader
    # workers fork from the main process and share these pages via
    # copy-on-write, so the cost is paid once (~6-7 GB for full DIV2K).
    cache_in_memory: bool = True
    # Variable LR degradation (defaults reproduce clean bicubic).
    downsample_kind: DownsampleKind = "bicubic"
    noise_sigma: float = 0.0  # max gaussian sigma (in [0,1] pixel units)
    jpeg_quality_range: tuple[int, int] = (95, 95)  # min, max (95,95 = off)


def _list_images(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def _resolve_split_files(cfg: DataConfig, split: Split) -> list[Path]:
    if split == "train":
        files = _list_images(cfg.data_root / "DIV2K_train_HR")
        if not files:
            raise FileNotFoundError(
                f"No PNGs in {cfg.data_root / 'DIV2K_train_HR'}. "
                "Run scripts/download_div2k.py first."
            )
        return files
    valid = _list_images(cfg.data_root / "DIV2K_valid_HR")
    if len(valid) < cfg.val_count + cfg.test_count:
        raise FileNotFoundError(
            f"Need {cfg.val_count + cfg.test_count} valid images, "
            f"found {len(valid)}. Run scripts/download_div2k.py first."
        )
    if split == "val":
        return valid[: cfg.val_count]
    return valid[cfg.val_count : cfg.val_count + cfg.test_count]


def _bicubic_downsample(hr: torch.Tensor, scale: int) -> torch.Tensor:
    # hr: (3, H, W) in [0, 1]. F.interpolate needs a batch dim.
    lr = F.interpolate(
        hr.unsqueeze(0),
        scale_factor=1.0 / scale,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    return lr.clamp_(0.0, 1.0)


_DOWNSAMPLE_MODES = ("bicubic", "bilinear", "area")


def _downsample(hr: torch.Tensor, scale: int, mode: str) -> torch.Tensor:
    """Downsample HR by `scale` with the given interpolation mode."""
    kwargs: dict = {"scale_factor": 1.0 / scale, "mode": mode}
    if mode in ("bicubic", "bilinear"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    lr = F.interpolate(hr.unsqueeze(0), **kwargs).squeeze(0)
    return lr.clamp_(0.0, 1.0)


def _maybe_add_noise(lr: torch.Tensor, sigma_max: float, rng: random.Random) -> torch.Tensor:
    if sigma_max <= 0.0:
        return lr
    sigma = rng.uniform(0.0, sigma_max)
    if sigma == 0.0:
        return lr
    return (lr + torch.randn_like(lr) * sigma).clamp_(0.0, 1.0)


def _maybe_jpeg(lr: torch.Tensor, q_min: int, q_max: int, rng: random.Random) -> torch.Tensor:
    if q_min >= 95 and q_max >= 95:
        return lr
    q = rng.randint(q_min, q_max)
    pil = TF.to_pil_image(lr.clamp(0.0, 1.0))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    with Image.open(buf) as im:
        return TF.to_tensor(im.convert("RGB"))


def _make_lr(
    hr: torch.Tensor,
    scale: int,
    kind: str,
    noise_sigma: float,
    jpeg_q_range: tuple[int, int],
    rng: random.Random,
) -> torch.Tensor:
    """Apply the (optionally stochastic) degradation pipeline HR -> LR."""
    mode = (
        rng.choice(_DOWNSAMPLE_MODES) if kind == "mixed" else "bicubic"
    )
    lr = _downsample(hr, scale, mode)
    lr = _maybe_add_noise(lr, noise_sigma, rng)
    lr = _maybe_jpeg(lr, jpeg_q_range[0], jpeg_q_range[1], rng)
    return lr


def _apply_sync_aug(hr: torch.Tensor, rng: random.Random) -> torch.Tensor:
    # Random hflip + random 0/90/180/270 rotation, applied identically per
    # pair (HR/LR pair is enforced by deriving LR after augmentation).
    if rng.random() < 0.5:
        hr = TF.hflip(hr)
    k = rng.randint(0, 3)
    if k:
        hr = torch.rot90(hr, k=k, dims=(-2, -1))
    return hr


def _load_uint8_tensor(path: Path) -> torch.Tensor:
    """Return (3, H, W) uint8 tensor from a PNG."""
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"))  # writable (H, W, 3) uint8
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class DIV2KDataset(Dataset):
    """Patch dataset for DIV2K x4 super-resolution.

    When `cfg.cache_in_memory=True`, all HR images are loaded once into a
    list of uint8 tensors and shared with workers via fork (copy-on-write
    on Linux). This removes the PNG-decode-per-sample bottleneck that
    otherwise leaves the GPU mostly idle.
    """

    def __init__(self, cfg: DataConfig, split: Split) -> None:
        self.cfg = cfg
        self.split = split
        self.files = _resolve_split_files(cfg, split)
        self.hr_size = cfg.hr_size
        self.scale = cfg.scale

        self._cache: list[torch.Tensor] | None = None
        if cfg.cache_in_memory:
            self._cache = []
            total_bytes = 0
            for p in tqdm(self.files, desc=f"[cache {split}]", leave=False):
                t = _load_uint8_tensor(p)
                self._cache.append(t)
                total_bytes += t.numel()
            mb = total_bytes / 1024**2
            print(f"[data] cached {len(self._cache)} {split} images in RAM ({mb:.0f} MiB).")
            self._sizes = [(t.shape[1], t.shape[2]) for t in self._cache]
        else:
            self._sizes = []
            for p in self.files:
                with Image.open(p) as im:
                    w, h = im.size
                self._sizes.append((h, w))

        if split == "train":
            self._length = len(self.files) * cfg.patches_per_train_image
            self._index: list[tuple[int, int, int]] | None = None
        else:
            per_image = (
                cfg.val_patches_per_image if split == "val" else cfg.test_patches_per_image
            )
            base_seed = cfg.seed + (0 if split == "val" else 10_000)
            rng = random.Random(base_seed)
            index: list[tuple[int, int, int]] = []
            for img_idx, (h, w) in enumerate(self._sizes):
                if w < self.hr_size or h < self.hr_size:
                    raise ValueError(
                        f"{self.files[img_idx].name} smaller than hr_size={self.hr_size}"
                    )
                for _ in range(per_image):
                    top = rng.randint(0, h - self.hr_size)
                    left = rng.randint(0, w - self.hr_size)
                    index.append((img_idx, top, left))
            self._index = index
            self._length = len(index)

    def __len__(self) -> int:
        return self._length

    def _get_hr_tensor(self, img_idx: int, top: int, left: int) -> torch.Tensor:
        if self._cache is not None:
            patch = self._cache[img_idx][
                :, top : top + self.hr_size, left : left + self.hr_size
            ]
            return patch.float().div_(255.0)
        with Image.open(self.files[img_idx]) as im:
            hr_pil = im.convert("RGB").crop(
                (left, top, left + self.hr_size, top + self.hr_size)
            )
        return TF.to_tensor(hr_pil)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split == "train":
            img_idx = idx % len(self.files)
            h, w = self._sizes[img_idx]
            rng = random.Random(self.cfg.seed * 1_000_003 + idx)
            top = rng.randint(0, h - self.hr_size)
            left = rng.randint(0, w - self.hr_size)
            hr = self._get_hr_tensor(img_idx, top, left)
            hr = _apply_sync_aug(hr, rng)
            lr = _make_lr(
                hr,
                scale=self.scale,
                kind=self.cfg.downsample_kind,
                noise_sigma=self.cfg.noise_sigma,
                jpeg_q_range=self.cfg.jpeg_quality_range,
                rng=rng,
            )
        else:
            assert self._index is not None
            img_idx, top, left = self._index[idx]
            hr = self._get_hr_tensor(img_idx, top, left)
            # Val/test: clean bicubic only, to keep eval comparable across runs.
            lr = _bicubic_downsample(hr, self.scale)

        return lr, hr


def get_dataloaders(
    cfg: DataConfig | None = None,
    batch_size: int = 16,
    num_workers: int = 8,
) -> dict[Split, DataLoader]:
    cfg = cfg or DataConfig()
    pin = torch.cuda.is_available()
    loaders: dict[Split, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = DIV2KDataset(cfg, split)  # type: ignore[arg-type]
        loaders[split] = DataLoader(  # type: ignore[index]
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=(split == "train"),
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    return loaders
