"""Generate low-res test images for the web app.

Reads DIV2K_valid_HR, crops patches at multiple sizes, downscales them to
simulate LR input. Output saved to outputs/test_inputs/.

Usage:
    python scripts/gen_test_images.py
    python scripts/gen_test_images.py --n 10 --sizes 64 128 256
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

VAL_DIR   = Path("data/DIV2K_valid_HR")
OUT_DIR   = Path("outputs/test_inputs")
SEED      = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",     type=int, default=6,
                   help="Number of test images to generate (default: 6)")
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256],
                   help="LR patch sizes in pixels (default: 64 128 256)")
    p.add_argument("--out",   type=Path, default=OUT_DIR)
    return p.parse_args()


def interesting_crop(img: Image.Image, size: int, rng: random.Random) -> Image.Image:
    """Random crop that avoids boring uniform corners."""
    w, h = img.size
    if w < size or h < size:
        img = img.resize((max(w, size), max(h, size)), Image.LANCZOS)
        w, h = img.size
    x = rng.randint(0, w - size)
    y = rng.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    images = sorted(VAL_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG found in {VAL_DIR}. Download DIV2K first.")

    rng = random.Random(SEED)
    chosen = rng.sample(images, min(args.n, len(images)))

    generated = []
    for img_path in chosen:
        hr = Image.open(img_path).convert("RGB")
        for size in args.sizes:
            crop = interesting_crop(hr, size, rng)
            name = f"{img_path.stem}_lr{size}.png"
            out_path = args.out / name
            crop.save(out_path)
            generated.append((name, size))

    print(f"Generated {len(generated)} test images → {args.out}/")
    print()
    for name, size in generated:
        print(f"  {name}  ({size}×{size} px → modelo dará {size*4}×{size*4})")
    print()
    print(f"Sube cualquiera de estas en http://localhost:8000")


if __name__ == "__main__":
    main()
