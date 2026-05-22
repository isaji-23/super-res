"""Evaluate external SOTA baselines on the DIV2K test split.

Runs the same PSNR/SSIM/LPIPS pipeline as `src/evaluate.py` but for
external pretrained baselines (EDSR-base, DRLN, Real-ESRGAN x4plus).
Writes per-baseline aggregate metrics to `outputs/baselines_metrics.json`
so the notebook can render a unified table v3 vs baselines.

Usage (from repo root):
    PYTHONPATH=. python scripts/eval_baselines.py

External weights are downloaded on first run:
  * EDSR-base, DRLN via Hugging Face Hub (super-image package).
  * Real-ESRGAN x4plus from xinntao's GitHub release (~64MB).

The test split uses the deterministic LR/HR patches DIV2KDataset
generates with `seed=1337` — identical to what `src/evaluate.py` uses
for v1/v2/v3, so the numbers are directly comparable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import DataConfig, DIV2KDataset
from src.metrics import lpips_metric, psnr, ssim
from src.realesrgan_arch import RRDBNet

_HF_BASELINES: dict[str, tuple[str, str, str]] = {
    "edsr": ("EdsrModel", "eugenesiow/edsr-base", "EDSR-base (DIV2K x4)"),
    "drln": ("DrlnModel", "eugenesiow/drln",      "DRLN (DIV2K x4)"),
}

_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
_REALESRGAN_CACHE = Path.home() / ".cache" / "realesrgan" / "RealESRGAN_x4plus.pth"


def _load_hf(class_name: str, hf_id: str, device: torch.device):
    try:
        import super_image
    except ImportError as e:
        print(f"[skip] super-image not installed ({e}) — {hf_id} skipped")
        return None
    cls = getattr(super_image, class_name, None)
    if cls is None:
        print(f"[skip] super-image has no class {class_name} — {hf_id} skipped")
        return None
    try:
        m = cls.from_pretrained(hf_id, scale=4)
    except Exception as e:
        print(f"[skip] could not load {hf_id}: {e}")
        return None
    return m.eval().to(device)


def _ensure_realesrgan_weights() -> Path | None:
    if _REALESRGAN_CACHE.exists():
        return _REALESRGAN_CACHE
    try:
        _REALESRGAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[dl] Real-ESRGAN weights -> {_REALESRGAN_CACHE}")
        torch.hub.download_url_to_file(_REALESRGAN_URL, str(_REALESRGAN_CACHE), progress=True)
        return _REALESRGAN_CACHE
    except Exception as e:
        print(f"[skip] Real-ESRGAN download failed: {e}")
        return None


def _load_realesrgan(device: torch.device):
    w = _ensure_realesrgan_weights()
    if w is None:
        return None
    try:
        m = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4,
                    num_feat=64, num_block=23, num_grow_ch=32)
        ckpt = torch.load(w, map_location=device, weights_only=False)
        state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        m.load_state_dict(state, strict=True)
        return m.eval().to(device)
    except Exception as e:
        print(f"[skip] Real-ESRGAN load failed: {e}")
        return None


@torch.no_grad()
def _eval(model, loader, device) -> dict[str, float]:
    p_sum = s_sum = l_sum = 0.0
    n = 0
    for lr, hr in tqdm(loader, desc="eval", leave=False):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        out = model(lr).clamp(0.0, 1.0)
        # Some baselines may produce slightly off-size outputs; crop.
        out = out[:, :, : hr.shape[2], : hr.shape[3]]
        p_sum += psnr(out, hr)
        s_sum += ssim(out, hr)
        l_sum += lpips_metric(out, hr)
        n += 1
    return {"psnr": p_sum / max(n, 1),
            "ssim": s_sum / max(n, 1),
            "lpips": l_sum / max(n, 1)}


def main() -> None:
    outputs_dir = _REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / "baselines_metrics.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    cfg = DataConfig()  # identical settings to src/evaluate.py
    ds = DIV2KDataset(cfg, "test")
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=2, pin_memory=device.type == "cuda")
    print(f"[data] test patches: {len(ds)}")

    results: dict[str, dict] = {}

    for key, (class_name, hf_id, label) in _HF_BASELINES.items():
        print(f"\n[load] {key}: {label}")
        m = _load_hf(class_name, hf_id, device)
        if m is None:
            continue
        print(f"[eval] {key}")
        metrics = _eval(m, loader, device)
        results[key] = {"label": label, **metrics}
        print(f"[done] {key}: PSNR={metrics['psnr']:.2f}  "
              f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[load] realesrgan: Real-ESRGAN x4plus")
    m = _load_realesrgan(device)
    if m is not None:
        print(f"[eval] realesrgan")
        metrics = _eval(m, loader, device)
        results["realesrgan"] = {"label": "Real-ESRGAN x4plus (xinntao)", **metrics}
        print(f"[done] realesrgan: PSNR={metrics['psnr']:.2f}  "
              f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}")

    if not results:
        print("\n[error] no baselines evaluated")
        sys.exit(1)

    with out_path.open("w") as f:
        json.dump({"split": "test", "baselines": results}, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
