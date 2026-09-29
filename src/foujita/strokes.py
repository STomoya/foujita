"""Stroke sequence container, `.npz` IO and canvas compositing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

# Polyline of round-capped segments, alpha 1. Points are (x, y) in pixel coordinates of the canvas,
# radius is in pixels, color is RGB float in [0, 1]. Array order is painting order.
CURVED_V1 = 'curved-polyline-v1'


@dataclass
class Strokes:
    """Ordered curved strokes. All lists share the same length and are in painting order."""

    canvas_size: tuple[int, int]  # (H, W)
    radius: list[float]
    color: list[np.ndarray]  # each (3,)
    points: list[np.ndarray]  # each (n, 2)
    format: str = CURVED_V1

    def __len__(self) -> int:
        return len(self.radius)

    def save(self, path: Path, **extra: float) -> None:
        """Write to npz. `extra` scalars (e.g. time_s) are stored alongside."""
        lens = np.array([len(p) for p in self.points])
        np.savez_compressed(
            path,
            format=self.format,
            canvas_size=np.array(self.canvas_size),
            radius=np.array(self.radius, dtype=np.float32),
            color=np.array(self.color, dtype=np.float32).reshape(-1, 3),
            point_offsets=np.concatenate([[0], np.cumsum(lens)]),
            points=np.concatenate(self.points).astype(np.float32) if self.points else np.zeros((0, 2), np.float32),
            **extra,
        )

    @classmethod
    def load(cls, path: Path) -> tuple[Strokes, dict[str, float]]:
        """Read an npz written by `save`. Returns the strokes and the extra scalars."""
        d = np.load(path)
        off, pts = d['point_offsets'], d['points']
        strokes = cls(
            canvas_size=tuple(int(v) for v in d['canvas_size']),
            radius=d['radius'].tolist(),
            color=list(d['color']),
            points=[pts[a:b] for a, b in pairwise(off)],
            format=str(d['format']),
        )
        known = {'format', 'canvas_size', 'radius', 'color', 'point_offsets', 'points'}
        return strokes, {k: float(d[k]) for k in d.files if k not in known}


def blank_canvas(size: tuple[int, int]) -> np.ndarray:
    """White (H, W, 3) float32 canvas."""
    return np.ones((*size, 3), dtype=np.float32)


def paint_stroke(canvas: np.ndarray, points: np.ndarray, radius: float, color: np.ndarray) -> None:
    """Composite one stroke onto `canvas` in place (anti-aliased, only touches the bounding box)."""
    h, w = canvas.shape[:2]
    x0, y0 = np.maximum(np.floor(points.min(0) - radius - 1), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(points.max(0) + radius + 2), [w, h]).astype(int)
    if x1 <= x0 or y1 <= y0:
        return
    p = np.stack(np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)), axis=-1).astype(np.float32)
    dist = np.full(p.shape[:2], np.inf, dtype=np.float32)
    for a, b in zip(points, points[1:] if len(points) > 1 else points, strict=False):
        d = b - a
        t = np.clip(((p - a) @ d) / max(float(d @ d), 1e-8), 0, 1)
        dist = np.minimum(dist, np.linalg.norm(p - (a + t[..., None] * d), axis=-1))
    alpha = np.clip(radius + 0.5 - dist, 0, 1)[..., None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - alpha) + color * alpha


def render(strokes: Strokes, upto: int | None = None) -> np.ndarray:
    """Render the first `upto` strokes (all by default) on a blank canvas."""
    canvas = blank_canvas(strokes.canvas_size)
    for i in range(len(strokes) if upto is None else upto):
        paint_stroke(canvas, strokes.points[i], strokes.radius[i], strokes.color[i])
    return canvas
