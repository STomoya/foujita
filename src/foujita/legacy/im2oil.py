"""Im2Oil (Tong et al., ACM MM 2022): texture-density adaptive sampling + Voronoi cells -> one stroke per cell."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms.v2.functional import gaussian_blur

from foujita.strokes import BRUSH_V1, CURVED_V1, PAINTERS, Strokes, blank_canvas


@dataclass
class Im2OilParams:
    """Fineness is linear in `density`: number of strokes = density * H * W."""

    density: float = 0.1  # anchors per pixel, the single fineness knob
    texture_weight: float = 4.0  # how strongly texture-rich areas attract anchors (0 = uniform)
    texture_sigma: float = 2.0  # blur of the gradient magnitude that forms the density map
    orient_sigma: float = 3.0  # blur of the structure tensor that gives stroke orientation
    lloyd_iters: int = 5  # density-weighted Lloyd relaxation steps
    length_factor: float = 2.0  # stroke length = length_factor * cell diameter
    width_factor: float = 1.0  # stroke radius = width_factor * cell radius
    textured: bool = True  # oil brush template (BRUSH_V1) instead of plain capsules (CURVED_V1)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    k = 2 * math.ceil(3 * sigma) + 1
    return gaussian_blur(torch.from_numpy(x)[None], [k, k], [sigma, sigma])[0].numpy()


def _assign(pix: torch.Tensor, anchors: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Index of the nearest anchor for every pixel (the Voronoi cells)."""
    return torch.cat([torch.cdist(c, anchors).argmin(1) for c in pix.split(chunk)])


def paint(image: np.ndarray, params: Im2OilParams, rng: np.random.Generator) -> Strokes:
    """Paint `image` ((H, W, 3) float32 in [0, 1]) and return the strokes in painting order (large cells first)."""
    h, w = image.shape[:2]
    lum = image @ np.array([0.3, 0.59, 0.11], dtype=np.float32)
    gy, gx = np.gradient(lum)
    texture = _blur(np.hypot(gx, gy).astype(np.float32), params.texture_sigma)
    dens = 1 + params.texture_weight * texture / max(float(texture.mean()), 1e-8)
    dens_flat = dens.ravel().astype(np.float64)

    # adaptive sampling: anchors drawn from the density map, then relaxed towards density-weighted centroids
    n = max(round(params.density * h * w), 1)
    idx = rng.choice(h * w, n, replace=False, p=dens_flat / dens_flat.sum())
    ys, xs = np.divmod(np.arange(h * w), w)
    xy = np.stack([xs, ys], axis=1).astype(np.float32)
    pix = torch.from_numpy(xy)
    anchors = torch.from_numpy(xy[idx])
    wt = torch.from_numpy(dens_flat.astype(np.float32))
    for _ in range(params.lloyd_iters):
        cell = _assign(pix, anchors)
        mass = torch.zeros(n).index_add_(0, cell, wt)
        cen = torch.zeros(n, 2).index_add_(0, cell, pix * wt[:, None])
        anchors = torch.where(mass[:, None] > 0, cen / mass.clamp(min=1e-8)[:, None], anchors)
    cell = _assign(pix, anchors)

    # per cell: mean color, size from area, orientation along the edge (perpendicular to the structure tensor gradient)
    count = torch.bincount(cell, minlength=n).float()
    color = torch.zeros(n, 3).index_add_(0, cell, torch.from_numpy(image.reshape(-1, 3))) / count.clamp(min=1)[:, None]
    jxx, jxy, jyy = (_blur(a.astype(np.float32), params.orient_sigma).ravel() for a in (gx * gx, gx * gy, gy * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # gradient direction + 90 deg
    anchor_px = anchors.round().long().clamp(min=0)
    ai = (anchor_px[:, 1].clamp(max=h - 1) * w + anchor_px[:, 0].clamp(max=w - 1)).numpy()
    theta = theta[ai]

    canvas = blank_canvas((h, w))
    out = Strokes(canvas_size=(h, w), radius=[], color=[], points=[], format=BRUSH_V1 if params.textured else CURVED_V1)
    draw = PAINTERS[out.format]
    diam = 2 * np.sqrt(count.numpy() / np.pi)  # cell diameter
    for i in np.argsort(-diam, kind='stable'):
        if count[i] == 0:
            continue
        half = params.length_factor * diam[i] / 2 * np.array([np.cos(theta[i]), np.sin(theta[i])])
        c = anchors[i].numpy()
        pts = np.clip([c - half, c + half], 0, [w - 1, h - 1]).astype(np.float32)
        r = max(params.width_factor * diam[i] / 2, 1.0)
        col = color[i].numpy()
        draw(canvas, pts, r, col)
        out.radius.append(float(r))
        out.color.append(col)
        out.points.append(pts)
    return out
