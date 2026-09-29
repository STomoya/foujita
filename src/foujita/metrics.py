"""Reconstruction metrics on (H, W, 3) float arrays in [0, 1]."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(((a - b) ** 2).mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB with peak 1."""
    return float(-10 * np.log10(max(mse(a, b), 1e-10)))


def ssim(a: np.ndarray, b: np.ndarray, window: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM with a Gaussian window (Wang et al. 2004), averaged over channels."""
    x = torch.from_numpy(a).permute(2, 0, 1)[None].double()
    y = torch.from_numpy(b).permute(2, 0, 1)[None].double()
    ax = torch.arange(window, dtype=torch.double) - (window - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(3, 1, window, window)

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, k, groups=3)

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return float((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))).mean())
