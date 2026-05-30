"""Open-loop ego-trajectory metrics for nuScenes evaluation.

Dependency-light on purpose: this module must NOT import the nuScenes dataset
(which pulls `nuscenes`/`pyquaternion` at import time) so the trainer can import it
on the hot path. The one angle helper is duplicated locally.

Inputs are the *denormalized* predicted/GT waypoint sequences as produced by the
action denormalization round-trip in the trainer/eval, i.e. shape ``[T, 3]`` where
each row is ``(dx_forward, dy_left, dyaw)`` cumulative waypoints in the anchor ego
frame, in meters / radians. These are the UniAD/VAD/ST-P3 open-loop conventions.
"""

from typing import Sequence

import numpy as np

try:  # torch is always present, but keep the numpy path import-safe.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _to_np(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().to(device="cpu", dtype=torch.float32).numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:  # [1, T, 3] -> [T, 3]
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected waypoints of shape [T, >=3], got {arr.shape}")
    return arr


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def open_loop_trajectory_metrics(
    pred_xytheta,
    gt_xytheta,
    action_hz: float = 2.0,
    horizons_s: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
) -> dict:
    """Compute open-loop trajectory metrics (meters / radians).

    Returns a dict with:
        ade        : mean over waypoints of ||pred_xy - gt_xy||
        fde        : ||pred_xy[-1] - gt_xy[-1]||
        l2@{h}s    : ||pred_xy[idx] - gt_xy[idx]|| for waypoint idx=round(h*hz)-1
        yaw_err    : mean |wrap(pred_yaw - gt_yaw)|
        num_waypoints : T

    Horizons whose waypoint index is out of range (T too small) are reported as NaN.
    Keys for horizons are formatted as e.g. "l2@1s", "l2@2s" (integer seconds when
    whole, else one decimal: "l2@1.5s").
    """
    pred = _to_np(pred_xytheta)
    gt = _to_np(gt_xytheta)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")

    T = pred.shape[0]
    step_l2 = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)  # [T]
    ade = float(step_l2.mean()) if T > 0 else float("nan")
    fde = float(step_l2[-1]) if T > 0 else float("nan")
    yaw_err = float(np.abs(_wrap_to_pi(pred[:, 2] - gt[:, 2])).mean()) if T > 0 else float("nan")

    out = {
        "ade": ade,
        "fde": fde,
        "yaw_err": yaw_err,
        "num_waypoints": int(T),
    }
    for h in horizons_s:
        idx = int(round(h * action_hz)) - 1  # 1s -> idx1, 2s -> idx3, ... at 2 Hz
        key = f"l2@{int(h)}s" if float(h).is_integer() else f"l2@{h:g}s"
        out[key] = float(step_l2[idx]) if 0 <= idx < T else float("nan")
    return out
