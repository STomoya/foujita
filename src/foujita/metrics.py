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


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


@cache
def _inception() -> torch.nn.Module:
    from pytorch_fid.inception import InceptionV3  # noqa: PLC0415

    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(_device())


def inception_features(images: np.ndarray, batch: int = 50) -> np.ndarray:
    """(N, H, W, 3) float images in [0, 1] -> (N, 2048) float64 pool features (pytorch-fid weights)."""
    net, out = _inception(), []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i : i + batch]).permute(0, 3, 1, 2).to(_device())
            out.append(net(x)[0].squeeze(-1).squeeze(-1).cpu().double().numpy())
    return np.concatenate(out)


def frechet_distance(m1: np.ndarray, s1: np.ndarray, m2: np.ndarray, s2: np.ndarray) -> float:
    """Frechet distance between two Gaussians."""
    from scipy import linalg  # noqa: PLC0415

    # pytorch-fid's own helper breaks on current scipy (sqrtm `disp` arg removed), so it is inlined here.
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():  # singular product: add a small ridge, as pytorch-fid does
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float((m1 - m2) @ (m1 - m2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(generated: list[np.ndarray], ref_mean: np.ndarray, ref_cov: np.ndarray) -> float:
    """FID of generated (H, W, 3) images against reference Inception statistics."""
    f = inception_features(np.stack(generated))
    return frechet_distance(f.mean(0), np.cov(f, rowvar=False), ref_mean, ref_cov)
