"""Fréchet Video Distance (FVD), set-level video-generation metric.

Two backends:
  - ``i3d``  (default, paper-comparable): the StyleGAN-V Kinetics-400 I3D TorchScript
    detector (`i3d_torchscript.pt`). NOT pip-installable / not auto-downloaded — pass a
    local path via the eval config. This is the de-facto FVD backbone used across the
    video-generation literature (StyleGAN-V, VideoGPT, etc.).
  - ``r3d``  (offline fallback): torchvision ``r3d_18`` pretrained on Kinetics-400. Ships
    with torchvision (no extra download), 512-d features. NOT paper-comparable — use only
    when the I3D weights are unavailable. The chosen backend is stamped into the metrics
    output so results are never silently mislabeled.

Input video tensors: ``[B, 3, T, H, W]`` in ``[0, 1]`` (T >= 9). FVD reuses
``frechet_distance``/``gaussian_stats`` from `fastwam.utils.perceptual_metrics`.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger
from fastwam.utils.perceptual_metrics import frechet_distance, gaussian_stats

logger = get_logger(__name__)

_KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
_KINETICS_STD = (0.22803, 0.22145, 0.216989)

_MODEL_CACHE: dict = {}


def load_fvd_model(backend: str = "i3d", i3d_weights: Optional[str] = None, device: Optional[torch.device] = None):
    """Load (and cache) the FVD feature backbone. Returns an opaque handle dict."""
    backend = str(backend).lower()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (backend, str(i3d_weights))
    if cache_key in _MODEL_CACHE:
        handle = _MODEL_CACHE[cache_key]
        handle["model"].to(device)
        handle["device"] = device
        return handle

    if backend == "i3d":
        if not i3d_weights:
            raise ValueError(
                "FVD backend 'i3d' requires `eval.fvd.i3d_weights` pointing to the "
                "StyleGAN-V `i3d_torchscript.pt` checkpoint. Provide it, or set "
                "`eval.fvd.backend=r3d` for the offline (not paper-comparable) fallback."
            )
        import os

        if not os.path.exists(i3d_weights):
            raise FileNotFoundError(f"I3D weights not found: {i3d_weights}")
        model = torch.jit.load(i3d_weights, map_location="cpu").to(device).eval()
        handle = {"backend": "i3d", "model": model, "device": device}
    elif backend == "r3d":
        from torchvision.models.video import r3d_18, R3D_18_Weights

        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = torch.nn.Identity()  # 512-d pre-fc features
        model = model.to(device).eval()
        handle = {"backend": "r3d", "model": model, "device": device}
    else:
        raise ValueError(f"Unknown FVD backend '{backend}'. Use 'i3d' or 'r3d'.")

    for p in handle["model"].parameters():
        p.requires_grad_(False)
    _MODEL_CACHE[cache_key] = handle
    return handle


@torch.no_grad()
def fvd_features(videos: torch.Tensor, handle: dict) -> torch.Tensor:
    """Extract FVD features for a batch of clips.

    `videos`: [B, 3, T, H, W] in [0, 1], T >= 9. Returns [B, D] on CPU (float32).
    """
    if videos.ndim != 5 or videos.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, T, H, W], got {tuple(videos.shape)}")
    if videos.shape[2] < 9:
        raise ValueError(f"FVD requires T >= 9 frames, got T={videos.shape[2]}")
    backend = handle["backend"]
    device = handle["device"]
    model = handle["model"]
    x = videos.to(device=device, dtype=torch.float32)

    if backend == "i3d":
        # I3D detector convention: [B, C, T, H, W] in [-1, 1], 224x224.
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, 224, 224]
        x = x * 2.0 - 1.0
        detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        feats = model(x, **detector_kwargs)
    else:  # r3d
        # torchvision video models: [B, C, T, H, W], Kinetics-normalized, 112x112.
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        mean = torch.tensor(_KINETICS_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_KINETICS_STD, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = x.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, 112, 112]
        feats = model(x)  # [B, 512]

    return feats.detach().to(device="cpu", dtype=torch.float32)


def fvd_from_features(real_feats: torch.Tensor, gen_feats: torch.Tensor) -> float:
    """FVD between two [N, D] feature sets (real vs generated clips)."""
    mu_r, sig_r = gaussian_stats(real_feats)
    mu_g, sig_g = gaussian_stats(gen_feats)
    return frechet_distance(mu_r, sig_r, mu_g, sig_g)
