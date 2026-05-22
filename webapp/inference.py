"""Model loading and inference for the super-resolution web app.

Serves project model versions side-by-side with an external SOTA baseline:
  * v1 = `src.model.UNetSR` (BN + sigmoid, original).
  * v2 = `src.model_v2.UNetSRv2` (no BN, no sigmoid, residual over bicubic).
  * v3 = `src.model_v3.UNetSRv3`.
  * edsr = `super_image.EdsrModel` (pretrained on DIV2K x4) — external
    reference to compare the project model against a strong baseline.
  * realesrgan = `RRDBNet` con pesos oficiales `RealESRGAN_x4plus.pth`
    (Real-ESRGAN, xinntao) — baseline SOTA para fotos reales.

All checkpoints are loaded at startup; `upscale_image` runs every available
model and returns the LR-nearest, bicubic and per-model PNGs so the frontend
can compare any pair via dropdowns.
"""
from __future__ import annotations

import base64
import io
import math
import os
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.model import UNetSR, count_parameters
from src.model_v2 import UNetSRv2
from src.model_v3 import UNetSRv3
from src.realesrgan_arch import RRDBNet

_ROOT = Path(__file__).parent.parent
_CHECKPOINTS: dict[str, Path] = {
    "v1": _ROOT / "checkpoints" / "best.pt",
    "v2": _ROOT / "checkpoints_v2" / "best.pt",
    "v3": _ROOT / "checkpoints_v3" / "best.pt",
}
_MODEL_CLASSES = {"v1": UNetSR, "v2": UNetSRv2, "v3": UNetSRv3}
_BASE_CHANNELS = 48
_MAX_INPUT = int(os.environ.get("SR_MAX_INPUT", 512))
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

_BASELINES: dict[str, tuple[str, str, str]] = {
    "edsr": ("EdsrModel", "eugenesiow/edsr-base", "EDSR-base (DIV2K x4)"),
    "drln": ("DrlnModel", "eugenesiow/drln",      "DRLN (DIV2K x4)"),
}
_BASELINE_KEYS = set(_BASELINES.keys())

_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
_REALESRGAN_CACHE = Path.home() / ".cache" / "realesrgan" / "RealESRGAN_x4plus.pth"
_REALESRGAN_LABEL = "Real-ESRGAN x4plus (xinntao)"

_models: dict[str, torch.nn.Module] = {}
_device: torch.device | None = None
_param_counts: dict[str, int] = {}


def _load_one(version: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    cls = _MODEL_CLASSES[version]
    model = cls(base_channels=_BASE_CHANNELS)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def _load_baseline(class_name: str, hf_id: str, device: torch.device) -> torch.nn.Module | None:
    """Load a pretrained baseline from Hugging Face via super-image.

    Returns None if the package or weights cannot be loaded (network down,
    dependency missing) so the rest of the app still starts."""
    try:
        import super_image
    except ImportError as e:
        print(f"[inference] super-image not installed ({e}) — {hf_id} skipped")
        return None

    cls = getattr(super_image, class_name, None)
    if cls is None:
        print(f"[inference] super-image has no class {class_name} — {hf_id} skipped")
        return None

    try:
        model = cls.from_pretrained(hf_id, scale=4)
    except Exception as e:
        print(f"[inference] failed to load baseline {hf_id}: {e}")
        return None

    model.eval().to(device)
    return model


def _ensure_realesrgan_weights() -> Path | None:
    """Download official RealESRGAN_x4plus.pth on first run; return path or None."""
    if _REALESRGAN_CACHE.exists():
        return _REALESRGAN_CACHE
    try:
        _REALESRGAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[inference] downloading Real-ESRGAN weights → {_REALESRGAN_CACHE}")
        torch.hub.download_url_to_file(_REALESRGAN_URL, str(_REALESRGAN_CACHE), progress=True)
        return _REALESRGAN_CACHE
    except Exception as e:
        print(f"[inference] Real-ESRGAN download failed: {e} — skipped")
        return None


def _load_realesrgan(device: torch.device) -> torch.nn.Module | None:
    weights = _ensure_realesrgan_weights()
    if weights is None:
        return None
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4,
                        num_feat=64, num_block=23, num_grow_ch=32)
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        # Real-ESRGAN releases use `params_ema`; fall back to `params` or raw dict.
        state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        return model
    except Exception as e:
        print(f"[inference] failed to load Real-ESRGAN: {e} — skipped")
        return None


def load_model() -> None:
    """Load every available checkpoint (project versions + external baseline).
    Missing checkpoints are skipped silently — the frontend will not offer
    that option."""
    global _device, _models, _param_counts

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _models = {}
    _param_counts = {}

    for version, path in _CHECKPOINTS.items():
        if not path.exists():
            print(f"[inference] checkpoint missing for {version}: {path} — skipped")
            continue
        m = _load_one(version, path, _device)
        _models[version] = m
        _param_counts[version] = count_parameters(m)[0]
        print(f"[inference] loaded {version} from {path} ({_param_counts[version]/1e6:.2f}M params)")

    for key, (class_name, hf_id, label) in _BASELINES.items():
        baseline = _load_baseline(class_name, hf_id, _device)
        if baseline is None:
            continue
        _models[key] = baseline
        _param_counts[key] = count_parameters(baseline)[0]
        print(
            f"[inference] loaded baseline {key} ({label}) from {hf_id} "
            f"({_param_counts[key]/1e6:.2f}M params)"
        )

    realesrgan = _load_realesrgan(_device)
    if realesrgan is not None:
        _models["realesrgan"] = realesrgan
        _param_counts["realesrgan"] = count_parameters(realesrgan)[0]
        print(
            f"[inference] loaded baseline realesrgan ({_REALESRGAN_LABEL}) "
            f"({_param_counts['realesrgan']/1e6:.2f}M params)"
        )

    if not _models:
        raise RuntimeError("No checkpoints available for any model version.")


def get_status() -> dict:
    return {
        "model_loaded": bool(_models),
        "device": str(_device) if _device else "none",
        "available": sorted(_models.keys()),
        "params_M": {k: round(v / 1e6, 2) for k, v in _param_counts.items()},
        "max_input_px": _MAX_INPUT,
    }


def _pad_to_multiple(img: Image.Image, multiple: int = 8) -> tuple[Image.Image, tuple[int, int]]:
    w, h = img.size
    pw = math.ceil(w / multiple) * multiple
    ph = math.ceil(h / multiple) * multiple
    if pw == w and ph == h:
        return img, (w, h)
    padded = Image.new("RGB", (pw, ph), (0, 0, 0))
    padded.paste(img, (0, 0))
    return padded, (w, h)


def _to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


def upscale_image(pil_img: Image.Image) -> dict:
    if not _models or _device is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    img = pil_img.convert("RGB")
    w, h = img.size

    if w > _MAX_INPUT or h > _MAX_INPUT:
        raise ValueError(
            f"Image {w}×{h} exceeds limit {_MAX_INPUT}px. "
            f"Resize to ≤{_MAX_INPUT}px on the longer side before uploading."
        )

    bicubic = img.resize((w * 4, h * 4), Image.BICUBIC)

    padded, (orig_w, orig_h) = _pad_to_multiple(img, multiple=8)
    tensor = TF.to_tensor(padded).unsqueeze(0).to(_device)         # [0,1]

    model_pngs: dict[str, str] = {}
    timings: dict[str, float] = {}
    for version, model in _models.items():
        # super-image baselines expect [0,1] input (their ImageLoader normalizes).
        # The rgb_range=255 in config is applied internally to the mean shift.
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out_cropped = out[:, :, : orig_h * 4, : orig_w * 4]
        sr_img = TF.to_pil_image(out_cropped.squeeze(0).clamp(0, 1).cpu())
        model_pngs[version] = _to_base64_png(sr_img)
        timings[version] = round(elapsed_ms, 1)

    lr_nearest = img.resize((w * 4, h * 4), Image.NEAREST)

    return {
        "lr_nearest_png": _to_base64_png(lr_nearest),
        "bicubic_png": _to_base64_png(bicubic),
        "model_v1_png": model_pngs.get("v1"),
        "model_v2_png": model_pngs.get("v2"),
        "model_v3_png": model_pngs.get("v3"),
        "model_edsr_png": model_pngs.get("edsr"),
        "model_drln_png": model_pngs.get("drln"),
        "model_realesrgan_png": model_pngs.get("realesrgan"),
        "available": sorted(_models.keys()),
        "original_size": [w, h],
        "output_size": [w * 4, h * 4],
        "inference_ms": timings,
        "device": str(_device),
    }


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path is None or not path.exists():
        print("Usage: python -m webapp.inference <image_path>")
        sys.exit(1)

    print("Loading models...")
    load_model()
    print(get_status())

    img = Image.open(path)
    result = upscale_image(img)
    for v in result["available"]:
        png_b64 = result[f"model_{v}_png"]
        out_path = path.with_stem(f"{path.stem}_sr_{v}")
        Image.open(io.BytesIO(base64.b64decode(png_b64))).save(out_path)
        print(f"Saved SR ({v}) → {out_path}  ({result['inference_ms'][v]} ms)")
    print(f"Size: {result['original_size']} → {result['output_size']}  device={result['device']}")
