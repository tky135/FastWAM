"""Perceptual / distribution video metrics: LPIPS (per-clip) and FID (set-level).

All heavy imports (`lpips`, `scipy`, torchvision Inception weights) are LAZY so that
training and offline environments never hard-depend on them. Input video tensors use
the same contract as `fastwam.utils.video_metrics`: shape ``[3, T, H, W]`` in ``[0, 1]``.

Note on FID: this uses torchvision's `inception_v3` (2048-d pre-fc pool features,
ImageNet-normalized 299x299 input), NOT the ported `pytorch-fid` weights. Absolute
values are therefore not bit-comparable to papers, but are internally consistent for
comparing checkpoints. FVD lives in `fastwam.utils.fvd` and reuses `frechet_distance`.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_LPIPS_MODEL = None
_LPIPS_NET = None
_INCEPTION_MODEL = None

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- LPIPS

def _get_lpips(net: str, device: torch.device):
    """Lazily build (and cache) the LPIPS network. Raises ImportError if `lpips`
    is not installed so callers can degrade gracefully."""
    global _LPIPS_MODEL, _LPIPS_NET
    if _LPIPS_MODEL is not None and _LPIPS_NET == net:
        return _LPIPS_MODEL.to(device)
    import lpips  # lazy; may raise ImportError

    model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _LPIPS_MODEL = model
    _LPIPS_NET = net
    return model


@torch.no_grad()
def lpips_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    device: Optional[torch.device] = None,
) -> float:
    """Mean LPIPS over frames. `pred`/`target`: [3, T, H, W] in [0, 1]."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_lpips(net, device)
    # [3, T, H, W] -> [T, 3, H, W], scale [0,1] -> [-1,1]
    p = pred.permute(1, 0, 2, 3).to(device=device, dtype=torch.float32) * 2.0 - 1.0
    t = target.permute(1, 0, 2, 3).to(device=device, dtype=torch.float32) * 2.0 - 1.0
    d = model(p, t)  # [T, 1, 1, 1]
    return float(d.mean().item())


# ----------------------------------------------------------------------- Inception/FID

def _get_inception(device: torch.device):
    global _INCEPTION_MODEL
    if _INCEPTION_MODEL is not None:
        return _INCEPTION_MODEL.to(device)
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = torch.nn.Identity()  # expose the 2048-d pre-logit pooled features
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _INCEPTION_MODEL = model
    return model


@torch.no_grad()
def inception_features(frames: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Extract 2048-d Inception features for a stack of frames.

    `frames`: [N, 3, H, W] in [0, 1]. Returns [N, 2048] on CPU (float32).
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_inception(device)
    x = frames.to(device=device, dtype=torch.float32)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    feats = model(x)  # [N, 2048]
    return feats.detach().to(device="cpu", dtype=torch.float32)


def gaussian_stats(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu [D], sigma [D, D]) for a feature matrix [N, D] (numpy float64)."""
    feats = features.detach().to(device="cpu", dtype=torch.float64).numpy()
    if feats.ndim != 2:
        raise ValueError(f"Expected [N, D] features, got {feats.shape}")
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, np.atleast_2d(sigma)


def gaussian_stats_from_sums(sum_x, sum_outer, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build (mu [D], sigma [D, D]) from streaming accumulators.

    Lets FVD/FID scale to the full set without all-gathering the feature matrix:
    each rank accumulates ``sum_x += feats.sum(0)``, ``sum_outer += feats.T @ feats``,
    ``n += feats.shape[0]`` (float64), and the small (sum_x, sum_outer, n) tuples are
    summed across ranks before this call.
    """
    sum_x = np.asarray(sum_x, dtype=np.float64)
    sum_outer = np.asarray(sum_outer, dtype=np.float64)
    n = int(n)
    if n < 2:
        raise ValueError(f"Need >= 2 samples for covariance, got n={n}.")
    mu = sum_x / n
    cov = (sum_outer - n * np.outer(mu, mu)) / (n - 1)
    return mu, np.atleast_2d(cov)


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6
) -> float:
    """Fréchet distance between two Gaussians (pytorch-fid formula). Lazy scipy import."""
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            logger.warning("frechet_distance: large imaginary component %g; taking real part.", m)
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def fid_from_features(real_feats: torch.Tensor, gen_feats: torch.Tensor) -> float:
    """FID between two [N, D] feature sets."""
    mu_r, sig_r = gaussian_stats(real_feats)
    mu_g, sig_g = gaussian_stats(gen_feats)
    return frechet_distance(mu_r, sig_r, mu_g, sig_g)
