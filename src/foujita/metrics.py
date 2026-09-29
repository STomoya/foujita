"""Reconstruction metrics on (H, W, 3) float arrays in [0, 1]."""

from __future__ import annotations

from functools import cache

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


@cache
def _lpips_net() -> torch.nn.Module:
    import lpips  # noqa: PLC0415

    return lpips.LPIPS(net='alex', verbose=False).eval()


def lpips_dist(a: np.ndarray, b: np.ndarray) -> float:
    """LPIPS (AlexNet), lower is closer."""
    x, y = (torch.from_numpy(v).permute(2, 0, 1)[None] * 2 - 1 for v in (a, b))
    with torch.no_grad():
        return float(_lpips_net()(x, y))


def _inception_stats(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from pytorch_fid.inception import InceptionV3  # noqa: PLC0415

    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), 50):
            batch = torch.from_numpy(np.stack(images[i : i + 50])).permute(0, 3, 1, 2)
            feats.append(net(batch)[0].squeeze(-1).squeeze(-1).numpy())
    f = np.concatenate(feats)
    return f.mean(0), np.cov(f, rowvar=False)


def fid(generated: list[np.ndarray], reference: list[np.ndarray]) -> float:
    """Frechet Inception Distance between two sets of (H, W, 3) images (pytorch-fid weights)."""
    from scipy import linalg  # noqa: PLC0415

    (m1, s1), (m2, s2) = _inception_stats(generated), _inception_stats(reference)
    # pytorch-fid's own helper breaks on current scipy (sqrtm `disp` arg removed), so it is inlined here.
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():  # singular product: add a small ridge, as pytorch-fid does
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float((m1 - m2) @ (m1 - m2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
