"""Evaluate trained UNetSR on DIV2K x4: metrics table + visual grids.

Compares the model against the bicubic baseline on the chosen split
(default: test). Computes PSNR / SSIM / LPIPS on every patch of the
split, dumps a metrics table (txt + json), and renders N visual
comparison grids [LR (nearest) | Bicubic | Model | HR] to
``outputs/examples/``.

CLI:
    python -m src.evaluate --checkpoint checkpoints/best.pt \\
        --num-examples 10 --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DataConfig, DIV2KDataset
from src.metrics import lpips_metric, psnr, ssim
from src.model import UNetSR
from src.model_v2 import UNetSRv2
from src.model_v3 import UNetSRv3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--num-examples", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def bicubic_upsample(lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Bicubic upscale baseline. Input/output in [0, 1]."""
    return F.interpolate(
        lr, scale_factor=scale, mode="bicubic", align_corners=False, antialias=True
    ).clamp_(0.0, 1.0)


def _load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    base_channels = int(args.get("base_channels", 48))
    version = args.get("model", "v1")
    model_cls = {"v1": UNetSR, "v2": UNetSRv2, "v3": UNetSRv3}.get(version, UNetSR)
    print(f"[ckpt] model={version} base_channels={base_channels}")
    model = model_cls(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def compute_full_metrics(
    model: UNetSR, loader: DataLoader, device: torch.device
) -> dict[str, dict[str, float]]:
    """Aggregate metrics for model vs bicubic baseline over the loader."""
    accum = {
        "model": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0},
        "bicubic": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0},
    }
    n_batches = 0
    for lr, hr in tqdm(loader, desc="eval"):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        pred = model(lr)
        bic = bicubic_upsample(lr)

        accum["model"]["psnr"] += psnr(pred, hr)
        accum["model"]["ssim"] += ssim(pred, hr)
        accum["model"]["lpips"] += lpips_metric(pred, hr)
        accum["bicubic"]["psnr"] += psnr(bic, hr)
        accum["bicubic"]["ssim"] += ssim(bic, hr)
        accum["bicubic"]["lpips"] += lpips_metric(bic, hr)
        n_batches += 1

    for k in accum:
        for m in accum[k]:
            accum[k][m] /= max(n_batches, 1)
    return accum


def _tensor_to_np(img: torch.Tensor):
    return img.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def render_examples(
    model: UNetSR,
    dataset: DIV2KDataset,
    n: int,
    out_dir: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    """Save N comparison grids and return per-example metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(dataset))
    rows = []
    for i in range(n):
        lr, hr = dataset[i]
        lr_b = lr.unsqueeze(0).to(device)
        hr_b = hr.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(lr_b)
        bic = bicubic_upsample(lr_b)
        # Nearest-upscaled LR is shown for size parity with the other panels.
        lr_nearest = F.interpolate(lr_b, scale_factor=4, mode="nearest")

        m = {
            "idx": i,
            "psnr_model": psnr(pred, hr_b),
            "ssim_model": ssim(pred, hr_b),
            "lpips_model": lpips_metric(pred, hr_b),
            "psnr_bicubic": psnr(bic, hr_b),
            "ssim_bicubic": ssim(bic, hr_b),
            "lpips_bicubic": lpips_metric(bic, hr_b),
        }
        rows.append(m)

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        panels = [
            ("LR (nearest x4)", lr_nearest[0]),
            (
                f"Bicubic\nPSNR={m['psnr_bicubic']:.2f}  SSIM={m['ssim_bicubic']:.3f}",
                bic[0],
            ),
            (
                f"Model\nPSNR={m['psnr_model']:.2f}  SSIM={m['ssim_model']:.3f}",
                pred[0],
            ),
            ("HR (target)", hr_b[0]),
        ]
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(_tensor_to_np(img))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"example_{i:02d}.png", dpi=120)
        plt.close(fig)
    return rows


def _format_table(metrics: dict[str, dict[str, float]]) -> str:
    header = f"{'method':<10} {'PSNR(dB)':>10} {'SSIM':>8} {'LPIPS':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for name in ("bicubic", "model"):
        row = metrics[name]
        lines.append(
            f"{name:<10} {row['psnr']:>10.2f} {row['ssim']:>8.4f} {row['lpips']:>8.4f}"
        )
    delta_psnr = metrics["model"]["psnr"] - metrics["bicubic"]["psnr"]
    delta_lpips = metrics["model"]["lpips"] - metrics["bicubic"]["lpips"]
    lines.append(sep)
    lines.append(
        f"delta      {delta_psnr:>+10.2f} {'':>8} {delta_lpips:>+8.4f}  "
        f"(model - bicubic)"
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.outputs_dir / "examples"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model, ckpt = _load_model(args.checkpoint, device)
    print(
        f"[ckpt] {args.checkpoint}  epoch={ckpt.get('epoch')}  "
        f"val_metrics={ckpt.get('val_metrics')}"
    )

    data_cfg = DataConfig(seed=args.seed)
    dataset = DIV2KDataset(data_cfg, args.split)  # type: ignore[arg-type]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    metrics = compute_full_metrics(model, loader, device)
    table = _format_table(metrics)
    print("\n" + table)

    (args.outputs_dir / "metrics_table.txt").write_text(table + "\n")
    with (args.outputs_dir / "metrics.json").open("w") as f:
        json.dump({"split": args.split, "metrics": metrics}, f, indent=2)

    print(f"\n[examples] rendering {args.num_examples} grids -> {examples_dir}")
    per_example = render_examples(
        model, dataset, args.num_examples, examples_dir, device
    )
    with (args.outputs_dir / "examples_metrics.json").open("w") as f:
        json.dump(per_example, f, indent=2)

    print("[done] evaluation complete.")


if __name__ == "__main__":
    main()
