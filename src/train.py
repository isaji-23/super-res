"""Training loop for UNetSR on DIV2K x4 super-resolution.

CLI:
    python -m src.train --epochs 50 --batch-size 16 --lr 1e-4 \\
        --checkpoint-dir checkpoints

Features:
  * Mixed precision (torch.cuda.amp) on CUDA.
  * Adam(0.9, 0.999) + CosineAnnealingLR.
  * Logs per epoch: train loss, val loss, val PSNR/SSIM/LPIPS.
  * Saves best-PSNR checkpoint to {checkpoint_dir}/best.pt and final to last.pt.
  * Plots curves to outputs/train_curves.png and dumps history.json.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data import DataConfig, get_dataloaders
from src.losses import CombinedLoss, LossWeights
from src.metrics import lpips_metric, psnr, ssim
from src.model import UNetSR, count_parameters
from src.model_v2 import UNetSRv2
from src.model_v3 import UNetSRv3
from src.utils import plot_curves, save_history_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--model", choices=("v1", "v2", "v3"), default="v1",
                   help="v1=original UNetSR (BN+sigmoid). v2=no BN, no sigmoid, residual over bicubic. "
                        "v3=v2 + RCAB (channel attention) + ICNR PixelShuffle init.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable in-memory HR cache (slower, lower RAM).")
    p.add_argument("--weight-l1", type=float, default=1.0)
    p.add_argument("--weight-perceptual", type=float, default=0.1)
    p.add_argument("--patches-per-image", type=int, default=16,
                   help="Train-only: virtual epoch length multiplier.")
    p.add_argument("--hr-size", type=int, default=128,
                   help="HR patch size (default 128).")
    p.add_argument("--degradation", choices=("bicubic", "mixed"), default="bicubic",
                   help="LR degradation kernel. 'mixed' = random per-sample bicubic/bilinear/area.")
    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="Max gaussian noise sigma on LR (0..1). 0 disables.")
    p.add_argument("--jpeg-min", type=int, default=95,
                   help="Lower bound of random JPEG quality on LR. 95 = effectively off.")
    p.add_argument("--jpeg-max", type=int, default=95,
                   help="Upper bound of random JPEG quality on LR.")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision (fp32).")
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after N epochs without val PSNR improvement. "
                        "0 disables early stopping (full schedule runs).")
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="Minimum PSNR improvement (dB) to reset the patience counter.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: UNetSR, loader, loss_fn: CombinedLoss, device: torch.device) -> dict[str, float]:
    model.eval()
    losses, psnrs, ssims, lpipss = [], [], [], []
    for lr, hr in loader:
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
        pred = model(lr)
        total, _ = loss_fn(pred, hr)
        losses.append(total.item())
        psnrs.append(psnr(pred, hr))
        ssims.append(ssim(pred, hr))
        lpipss.append(lpips_metric(pred, hr))
    return {
        "loss": sum(losses) / len(losses),
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "lpips": sum(lpipss) / len(lpipss),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"[device] {device}  amp={use_amp}")

    data_cfg = DataConfig(
        patches_per_train_image=args.patches_per_image,
        seed=args.seed,
        cache_in_memory=not args.no_cache,
        hr_size=args.hr_size,
        downsample_kind=args.degradation,
        noise_sigma=args.noise_sigma,
        jpeg_quality_range=(args.jpeg_min, args.jpeg_max),
    )
    loaders = get_dataloaders(data_cfg, batch_size=args.batch_size, num_workers=args.num_workers)

    model_cls = {"v1": UNetSR, "v2": UNetSRv2, "v3": UNetSRv3}[args.model]
    model = model_cls(base_channels=args.base_channels).to(device)
    total_p, trainable_p = count_parameters(model)
    print(f"[model] {args.model} base_channels={args.base_channels}  params={total_p/1e6:.2f}M (trainable {trainable_p/1e6:.2f}M)")

    loss_fn = CombinedLoss(
        LossWeights(l1=args.weight_l1, perceptual=args.weight_perceptual)
    ).to(device)
    print(f"[loss] base=l1 w_l1={args.weight_l1} w_perceptual={args.weight_perceptual}")
    print(
        f"[data] hr_size={args.hr_size} degradation={args.degradation} "
        f"noise_sigma_max={args.noise_sigma} jpeg=[{args.jpeg_min}, {args.jpeg_max}]"
    )
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "val_lpips": [], "lr": [],
    }
    best_psnr = -float("inf")
    epochs_since_improvement = 0
    stopped_early_at: int | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        t0 = time.time()
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch:03d}/{args.epochs}", leave=False)
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                pred = model(lr_imgs)
                total, _ = loss_fn(pred, hr_imgs)

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = lr_imgs.size(0)
            running += total.item() * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{running / n_seen:.4f}")

        train_loss = running / max(n_seen, 1)
        val_stats = evaluate(model, loaders["val"], loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_stats["loss"])
        history["val_psnr"].append(val_stats["psnr"])
        history["val_ssim"].append(val_stats["ssim"])
        history["val_lpips"].append(val_stats["lpips"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"[epoch {epoch:03d}/{args.epochs}] {elapsed:5.1f}s  "
            f"train={train_loss:.4f}  val={val_stats['loss']:.4f}  "
            f"PSNR={val_stats['psnr']:.2f}dB  SSIM={val_stats['ssim']:.4f}  "
            f"LPIPS={val_stats['lpips']:.4f}  lr={history['lr'][-1]:.2e}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_stats,
        }
        torch.save(ckpt, args.checkpoint_dir / "last.pt")
        if val_stats["psnr"] > best_psnr + args.min_delta:
            best_psnr = val_stats["psnr"]
            epochs_since_improvement = 0
            torch.save(ckpt, args.checkpoint_dir / "best.pt")
            print(f"  -> new best PSNR={best_psnr:.2f} dB, saved best.pt")
        else:
            epochs_since_improvement += 1
            if args.patience > 0:
                print(f"  -> no improvement ({epochs_since_improvement}/{args.patience})")

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            stopped_early_at = epoch
            print(
                f"[early-stop] PSNR no mejora desde hace {args.patience} epochs "
                f"(best={best_psnr:.2f} dB). Deteniendo en epoch {epoch}/{args.epochs}."
            )
            break

    plot_curves(history, args.outputs_dir / "train_curves.png")
    save_history_json(history, args.outputs_dir / "train_history.json")
    tail = (
        f" (early-stopped at epoch {stopped_early_at}/{args.epochs})"
        if stopped_early_at is not None else ""
    )
    print(f"[done] best val PSNR={best_psnr:.2f} dB{tail}. Curves -> {args.outputs_dir/'train_curves.png'}")


if __name__ == "__main__":
    main()
