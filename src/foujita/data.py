"""Image folder loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def list_images(path: Path) -> list[Path]:
    """Sorted image files of a directory (or the file itself)."""
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
