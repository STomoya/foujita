"""Painterly Rendering with Curved Brush Strokes of Multiple Sizes (Hertzmann, SIGGRAPH 1998)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms.v2.functional import gaussian_blur

from foujita.strokes import Strokes, blank_canvas, paint_stroke

_EPS = 1e-8


@dataclass
class HertzmannParams:
    """Paper parameters. Distances are Euclidean RGB in [0, 1], radii in pixels."""

    radii: tuple[float, ...] = (8, 4, 2)  # brush sizes, painted coarse to fine
    threshold: float = 0.1  # T: grid cell is repainted if mean error exceeds this
    blur_factor: float = 0.5  # f_sigma: reference blur = f_sigma * R
    grid_size: float = 1.0  # f_g: grid spacing = f_g * R
    curvature: float = 1.0  # f_c: 1 follows the edge normal fully, 0 goes straight
    min_length: int = 4
    max_length: int = 16


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    k = 2 * math.ceil(3 * sigma) + 1
    t = torch.from_numpy(image).permute(2, 0, 1)
    return gaussian_blur(t, [k, k], [sigma, sigma]).permute(1, 2, 0).numpy()


def _make_stroke(  # noqa: PLR0917
    x0: int,
    y0: int,
    r: float,
    ref: np.ndarray,
    canvas: np.ndarray,
    grad: tuple,
    p: HertzmannParams,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = ref.shape[:2]
    color = ref[y0, x0].copy()
    pts = [(float(x0), float(y0))]
    x, y = x0, y0
    last = np.zeros(2)
    for i in range(1, p.max_length + 1):
        xi, yi = round(x), round(y)
        if i > p.min_length and np.linalg.norm(ref[yi, xi] - canvas[yi, xi]) < np.linalg.norm(ref[yi, xi] - color):
            break
        gx, gy = grad[0][yi, xi], grad[1][yi, xi]
        if math.hypot(gx, gy) < _EPS:  # no gradient, no direction
            break
        d = np.array([-gy, gx])  # normal to gradient = along the edge
        if last @ d < 0:
            d = -d
        d = p.curvature * d + (1 - p.curvature) * last
        d = d / np.linalg.norm(d)
        x, y = x + r * d[0], y + r * d[1]
        if not (0 <= round(x) < w and 0 <= round(y) < h):
            break
        pts.append((x, y))
        last = d
    return color, np.array(pts, dtype=np.float32)


def paint(image: np.ndarray, params: HertzmannParams, rng: np.random.Generator) -> Strokes:
    """Paint `image` ((H, W, 3) float32 in [0, 1]) and return the strokes in painting order."""
    h, w = image.shape[:2]
    canvas = blank_canvas((h, w))
    out = Strokes(canvas_size=(h, w), radius=[], color=[], points=[])
    for r in sorted(params.radii, reverse=True):
        ref = _blur(image, params.blur_factor * r)
        lum = ref @ np.array([0.3, 0.59, 0.11], dtype=np.float32)
        grad = (np.gradient(lum, axis=1), np.gradient(lum, axis=0))
        diff = np.linalg.norm(canvas - ref, axis=2)
        g = max(round(params.grid_size * r), 1)
        layer = []
        for y in range(0, h, g):
            for x in range(0, w, g):
                cell = diff[y : y + g, x : x + g]
                if cell.mean() > params.threshold:
                    dy, dx = np.unravel_index(cell.argmax(), cell.shape)
                    layer.append(_make_stroke(x + dx, y + dy, r, ref, canvas, grad, params))
        for i in rng.permutation(len(layer)):  # random order avoids grid artefacts
            color, pts = layer[i]
            paint_stroke(canvas, pts, r, color)
            out.radius.append(r)
            out.color.append(color)
            out.points.append(pts)
    return out
