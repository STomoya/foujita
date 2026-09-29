"""Image folder loading."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def list_images(path: Path) -> list[Path]:
    """Image files of a directory (sorted), a single image, or a `.txt` list (one path per line, relative to it)."""
    if path.suffix == '.txt':
        return [path.parent / line for line in path.read_text().split() if line]
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob('*') if p.suffix.lower() in EXTS)


def load_image(path: Path, size: int) -> np.ndarray:
    """Load as RGB, center-crop to square, resize to (size, size). Returns (H, W, 3) float32 in [0, 1]."""
    img = Image.open(path).convert('RGB')
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255


def save_image(path: Path, canvas: np.ndarray) -> None:
    """Write an (H, W, 3) float canvas as png."""
    Image.fromarray((canvas.clip(0, 1) * 255).round().astype(np.uint8)).save(path)


def write_sample_list(src: Path, out: Path, n: int, seed: int, subdirs: list[str] | None = None) -> None:
    """Pick `n` images (seeded, sorted) from `src` (optionally only `src/<subdir>`s) and write a list file."""
    roots = [src / d for d in subdirs] if subdirs else [src]
    pool = sorted({p for r in roots for p in list_images(r)})
    if len(pool) < n:
        msg = f'only {len(pool)} images under {roots}, need {n}'
        raise ValueError(msg)
    picked = sorted(np.random.default_rng(seed).choice(len(pool), n, replace=False))
    out.write_text(''.join(f'{os.path.relpath(pool[i], out.parent)}\n' for i in picked))


def prepare_mnist(root: Path, out: Path, n: int) -> None:
    """Download MNIST to `root` and write the first `n` test digits as png into `out`."""
    from torchvision.datasets import MNIST  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    ds = MNIST(str(root), train=False, download=True)
    for i in range(n):
        ds[i][0].save(out / f'{i:05d}.png')
