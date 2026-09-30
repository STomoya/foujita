"""Per-group Inception statistics for FID, stored so any subset of groups can be combined exactly.

Each group (e.g. a WikiArt style) keeps its count, mean and centered scatter matrix `M2 = sum (x - mean)(x - mean)^T`.
Groups merge with the parallel-variance formula, so "all styles" is just the merge of every style file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from foujita.data import square_resize
from foujita.metrics import inception_features

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from PIL import Image

Stats = tuple[int, np.ndarray, np.ndarray]  # (n, mean (D,), m2 (D, D))


def build(rows: Iterable[tuple[str, Image.Image]], sizes: list[int], batch: int = 64) -> dict[int, dict[str, Stats]]:
    """Stream (group, image) pairs and accumulate statistics per size and group.

    Each image is center-cropped and resized to every size (the same preprocessing as generated canvases).
    """
    acc: dict[int, dict[str, list]] = {s: {} for s in sizes}  # size -> group -> [n, sum, sum of outer products]
    pending: list[tuple[str, Image.Image]] = []

    def flush() -> None:
        for size in sizes:
            imgs = np.stack([square_resize(im, size) for _, im in pending])
            feats = inception_features(imgs)
            for g in {g for g, _ in pending}:
                f = feats[[i for i, (gi, _) in enumerate(pending) if gi == g]]
                n, s1, s2 = acc[size].setdefault(g, [0, 0.0, 0.0])
                acc[size][g] = [n + len(f), s1 + f.sum(0), s2 + f.T @ f]
        pending.clear()

    for row in rows:
        pending.append(row)
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    out: dict[int, dict[str, Stats]] = {}
    for size, groups in acc.items():
        out[size] = {}
        for g, (n, s1, s2) in groups.items():
            mean = s1 / n
            out[size][g] = (n, mean, s2 - n * np.outer(mean, mean))
    return out


def merge(parts: list[Stats]) -> tuple[np.ndarray, np.ndarray]:
    """Combine group statistics into (mean, covariance) of their union."""
    n, mean, m2 = parts[0]
    for nb, mb, m2b in parts[1:]:
        d = mb - mean
        m2 = m2 + m2b + np.outer(d, d) * (n * nb / (n + nb))
        mean = mean + d * (nb / (n + nb))
        n += nb
    return mean, m2 / (n - 1)


def save(out: Path, stats: dict[str, Stats], meta: dict) -> None:
    """Write one npz per group (M2 as float32 to keep files ~17 MB) and `manifest.json`."""
    out.mkdir(parents=True, exist_ok=True)
    for g, (n, mean, m2) in stats.items():
        np.savez_compressed(out / f'{g}.npz', n=n, mean=mean, m2=m2.astype(np.float32))
    (out / 'manifest.json').write_text(json.dumps({**meta, 'counts': {g: s[0] for g, s in stats.items()}}, indent=2))


def load(path: Path, groups: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load and merge the given groups (all if None) into (mean, covariance)."""
    files = [path / f'{g}.npz' for g in groups] if groups else sorted(path.glob('*.npz'))
    parts = []
    for f in files:
        if not f.exists():
            msg = f'no FID stats {f}; available: {sorted(p.stem for p in path.glob("*.npz"))}'
            raise SystemExit(msg)
        d = np.load(f)
        parts.append((int(d['n']), d['mean'], d['m2'].astype(np.float64)))
    return merge(parts)
