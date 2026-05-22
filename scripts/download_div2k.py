"""Download and extract DIV2K HR train/valid splits if missing.

Usage:
    python scripts/download_div2k.py [--data-root data]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DIV2K_URLS = {
    "DIV2K_train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}
EXPECTED_COUNTS = {"DIV2K_train_HR": 800, "DIV2K_valid_HR": 100}


def _png_count(folder: Path) -> int:
    return len(list(folder.glob("*.png"))) if folder.exists() else 0


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100 / total_size)
        sys.stdout.write(
            f"\r  {downloaded / 1024**2:8.1f} MiB / {total_size / 1024**2:8.1f} MiB ({pct:5.1f}%)"
        )
        sys.stdout.flush()


def download_split(name: str, url: str, data_root: Path) -> None:
    target_dir = data_root / name
    if _png_count(target_dir) >= EXPECTED_COUNTS[name]:
        print(f"[skip] {name}: {_png_count(target_dir)} PNGs already present.")
        return

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / f"{name}.zip"
    if not zip_path.exists():
        print(f"[download] {url}")
        urlretrieve(url, zip_path, reporthook=_progress)
        print()

    print(f"[extract] {zip_path.name} -> {data_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_root)

    count = _png_count(target_dir)
    if count < EXPECTED_COUNTS[name]:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_COUNTS[name]} PNGs, found {count}."
        )
    print(f"[ok] {name}: {count} PNGs.")
    zip_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    for name, url in DIV2K_URLS.items():
        download_split(name, url, args.data_root)


if __name__ == "__main__":
    main()
