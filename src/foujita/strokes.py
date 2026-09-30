"""Stroke sequence container, `.npz` IO and canvas compositing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

# Polyline of round-capped segments, alpha 1. Points are (x, y) in pixel coordinates of the canvas,
# radius is in pixels, color is RGB float in [0, 1]. Array order is painting order.
CURVED_V1 = 'curved-polyline-v1'
# Same stroke layout as CURVED_V1 but always 2 points: the brush template `assets/brush-0.png` (Im2Oil, CC0) is
# stretched so its x axis runs from p0 to p1 (plus a radius of overhang at both ends) and its y axis spans the
# width 2 * radius. Template mask = alpha, template streaks modulate the color multiplicatively.
BRUSH_V1 = 'textured-brush-v1'
_BRUSH = Path(__file__).parent / 'assets' / 'brush-0.png'
_TEXTURE_STRENGTH = 4.0  # template streaks are faint (+-8%); amplify so they show
_EPS = 1e-3


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
        arrays: dict[str, Any] = {
            'format': self.format,
            'canvas_size': np.array(self.canvas_size),
            'radius': np.array(self.radius, dtype=np.float32),
            'color': np.array(self.color, dtype=np.float32).reshape(-1, 3),
            'point_offsets': np.concatenate([[0], np.cumsum(lens)]),
            'points': np.concatenate(self.points).astype(np.float32) if self.points else np.zeros((0, 2), np.float32),
            **extra,
        }
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> tuple[Strokes, dict[str, float]]:
        """Read an npz written by `save`. Returns the strokes and the extra scalars."""
        d = np.load(path)
        off, pts = d['point_offsets'], d['points']
        strokes = cls(
            canvas_size=(int(d['canvas_size'][0]), int(d['canvas_size'][1])),
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


@cache
def _brush(ky: int, kx: int) -> torch.Tensor:
    """Brush template as (1, 2, h, w): [alpha, shade * alpha], box-filtered by (ky, kx) to avoid aliasing."""
    t = torch.from_numpy(np.array(Image.open(_BRUSH), dtype=np.float32) / 255)
    alpha = ((t - 0.1) / 0.5).clamp(0, 1)
    inside = t[alpha >= 1]
    shade = torch.where(alpha >= 1, 1 + _TEXTURE_STRENGTH * (t / inside.mean() - 1), 1).clamp(
        0.3, 1.3
    )  # soft edge: no shading
    x = torch.stack([alpha, shade * alpha])[None]
    return F.avg_pool2d(x, (ky, kx), stride=(ky, kx), ceil_mode=True) if (ky, kx) != (1, 1) else x


def paint_brush(canvas: np.ndarray, points: np.ndarray, radius: float, color: np.ndarray) -> None:
    """Composite one textured brush stroke (`BRUSH_V1`) onto `canvas` in place."""
    h, w = canvas.shape[:2]
    mid = points.mean(0)
    d = points[-1] - points[0]
    length = float(np.hypot(*d)) + 2 * radius
    u = d / np.hypot(*d) if np.hypot(*d) > _EPS else np.array([1.0, 0.0])
    n = np.array([-u[1], u[0]])
    half = float(np.hypot(length / 2, radius)) + 1
    x0, y0 = np.maximum(np.floor(mid - half), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(mid + half) + 1, [w, h]).astype(int)
    if x1 <= x0 or y1 <= y0:
        return
    p = np.stack(np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)), axis=-1).astype(np.float32) - mid
    # template coords in [-1, 1]; grid_sample pads with zeros outside the brush
    grid = torch.from_numpy(np.stack([p @ u / (length / 2), p @ n / radius], axis=-1).astype(np.float32))[None]
    tw, th = 355, 298  # full-size template; box filter down to about the painted size
    tmpl = _brush(max(th // max(round(2 * radius), 1), 1), max(tw // max(round(length), 1), 1))
    a, sa = F.grid_sample(tmpl, grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0].numpy()
    shade = np.where(a > _EPS, sa / np.maximum(a, _EPS), 1.0)[..., None]
    a = a[..., None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a) + np.clip(color * shade, 0, 1) * a


PAINTERS = {CURVED_V1: paint_stroke, BRUSH_V1: paint_brush}


def render(strokes: Strokes, upto: int | None = None, size: int | None = None) -> np.ndarray:
    """Render the first `upto` strokes (all by default) on a blank canvas.

    With `size`, the vector strokes are scaled to a square `size` canvas (no upscaling blur).
    """
    h, w = strokes.canvas_size
    sx, sy = (size / w, size / h) if size else (1.0, 1.0)
    canvas = blank_canvas((size, size) if size else (h, w))
    draw = PAINTERS[strokes.format]
    for i in range(len(strokes) if upto is None else upto):
        draw(canvas, strokes.points[i] * [sx, sy], strokes.radius[i] * sx, strokes.color[i])
    return canvas
