"""Standalone nuScenes evaluation for a trained FastWAM checkpoint.

Scores both modalities over a (seeded, optionally subset) slice of the val/test split,
sharded across GPUs:

  Action (open-loop): ADE, FDE, L2@{1,2,3,4}s, yaw error  (meters / radians)
  Video:              PSNR, SSIM (rollout-vs-gt, rollout-vs-vae, vae-vs-gt), LPIPS,
                      FID (set-level), FVD (set-level)

The video branch is NOT action-conditioned (`action_conditioned: false`), so this is
open-loop by construction — we pass `action=None` to `model.infer` (no GT-trajectory
leak). FVD/FID are aggregated from streaming Gaussian sufficient statistics so the run
scales to the full set without all-gathering the feature matrices.

Run:
    accelerate launch --num_processes 2 experiments/nuscenes/eval_nuscenes.py \
        ckpt=<run>/checkpoints/weights/step_XXXXXX.pt \
        eval.num_samples=512 eval.fvd.i3d_weights=/path/to/i3d_torchscript.pt
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastwam.utils.action_metrics import open_loop_trajectory_metrics
from fastwam.utils.logging_config import get_logger
from fastwam.utils.perceptual_metrics import (
    frechet_distance,
    gaussian_stats_from_sums,
    inception_features,
)
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from fastwam.utils import fvd as fvd_mod

logger = get_logger(__name__)

for _name, _fn in (("eval", eval), ("max", lambda x: max(x)), ("split", lambda s, idx: s.split("/")[int(idx)])):
    try:
        OmegaConf.register_new_resolver(_name, _fn)
    except Exception:
        pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ACTION_KEYS = ["ade", "fde", "l2@1s", "l2@2s", "l2@3s", "l2@4s", "yaw_err"]
VIDEO_SCALAR_KEYS = ["psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg", "lpips"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(key, torch.bfloat16)


def _resolve_device(cfg: DictConfig) -> str:
    dev = cfg.eval.get("device")
    if dev is not None:
        return str(dev)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    candidates = []
    explicit = cfg.eval.get("dataset_stats_path")
    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))
    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")
    seen = set()
    for path in candidates:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.exists():
            return rp
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried eval.dataset_stats_path and the "
        "checkpoint's parent directories. Pass eval.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _denorm_action(processor, raw_action: torch.Tensor, proprio_btd: torch.Tensor) -> torch.Tensor:
    """Normalized action [T,3]/[1,T,3] -> denormalized [1,T,3] (meters/radians), via the
    same merger/normalizer round-trip the trainer uses. `proprio_btd` is [1,Tp,4] (along
    for the ride; only the action normalizer params affect the action output)."""
    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]
    if raw_action.ndim == 2:
        action_btd = raw_action.unsqueeze(0)
    elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
        action_btd = raw_action
    else:
        raise ValueError(f"action must be [T,D] or [1,T,D], got {tuple(raw_action.shape)}")
    action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
    batch = {"action": action_btd, "state": proprio_btd}
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged = {
        "action": {m["key"]: batch["action"][m["key"]].squeeze(0) for m in action_meta},
        "state": {m["key"]: batch["state"][m["key"]].squeeze(0) for m in state_meta},
    }
    merged = processor.action_state_merger.forward(merged)
    return merged["action"].unsqueeze(0)  # [1, T, 3]


def _accumulate(acc, feats: np.ndarray):
    """Streaming Gaussian sufficient statistics: acc = [sum_x[D], sum_outer[D,D], n]."""
    n, D = feats.shape
    if acc is None:
        acc = [np.zeros(D, np.float64), np.zeros((D, D), np.float64), 0]
    acc[0] += feats.sum(axis=0)
    acc[1] += feats.T @ feats
    acc[2] += int(n)
    return acc


def _combine(accs):
    accs = [a for a in accs if a is not None]
    if not accs:
        return None
    D = accs[0][0].shape[0]
    s = np.zeros(D, np.float64)
    o = np.zeros((D, D), np.float64)
    n = 0
    for a in accs:
        s += a[0]
        o += a[1]
        n += a[2]
    return [s, o, n]


def _frechet_from_accs(real_acc, gen_acc) -> Optional[float]:
    if real_acc is None or gen_acc is None or real_acc[2] < 2 or gen_acc[2] < 2:
        return None
    mu_r, sig_r = gaussian_stats_from_sums(*real_acc)
    mu_g, sig_g = gaussian_stats_from_sums(*gen_acc)
    return frechet_distance(mu_r, sig_r, mu_g, sig_g)


def _nanmean(rows, key) -> Optional[float]:
    vals = [r[key] for r in rows if key in r and r[key] is not None and r[key] == r[key]]
    return float(np.mean(vals)) if vals else None


def _save_sample(out_dir: Path, idx: int, pred_vt, vae_vt, gt_vt, pred_denorm, gt_denorm, val_ds, fps: int):
    tag = f"sample_{idx:06d}"
    parts = [pred_vt] + ([vae_vt] if vae_vt is not None else []) + [gt_vt]
    stitched = torch.cat(parts, dim=2)  # concat along H (pred | vae | gt, top-to-bottom)
    frames = []
    for t in range(stitched.shape[1]):
        fr = (stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        frames.append(Image.fromarray(fr))
    vdir = out_dir / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    save_mp4(frames, str(vdir / f"{tag}.mp4"), fps=fps)
    bevdir = out_dir / "bev"
    bevdir.mkdir(parents=True, exist_ok=True)
    viz = getattr(val_ds, "save_eval_visualization", None)
    if callable(viz):
        try:
            viz(eval_dir=str(bevdir), step_tag=tag, pred_action_denorm=pred_denorm, gt_action_denorm=gt_denorm)
        except Exception as exc:
            logger.warning("save_eval_visualization failed for %s: %r", tag, exc)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_nuscenes.yaml")
def main(cfg: DictConfig):
    t0 = time.time()
    state = PartialState()
    rank, world, is_main = state.process_index, state.num_processes, state.is_main_process

    if cfg.get("ckpt") is None:
        raise ValueError("cfg.ckpt must be set (path to a trained weights checkpoint).")
    seed = int(cfg.eval.get("seed", 42))
    set_global_seed(seed, get_worker_init_fn=False)

    device = _resolve_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(device).eval()

    stats_path = _resolve_dataset_stats_path(cfg)
    if is_main:
        logger.info("Using dataset stats: %s", stats_path)
    val_ds = instantiate(cfg.data.val, pretrained_norm_stats=str(stats_path))
    processor = val_ds.processor

    # ---- subset selection + sharding ----
    N = len(val_ds)
    num = cfg.eval.get("num_samples", 512)
    num = N if (num is None or int(num) <= 0 or int(num) > N) else int(num)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))[:num].tolist()
    my_indices = perm[rank::world] if world > 1 else perm
    if is_main:
        logger.info("Evaluating %d/%d samples across %d rank(s) (this rank: %d).", num, N, world, len(my_indices))
        if num < 256:
            logger.warning("num_samples=%d < 256: FVD/FID are unstable at small N.", num)

    tiled = bool(cfg.eval.get("tiled", True))
    compute_vae = bool(cfg.eval.get("compute_vae_recon", True))
    compute_fid = bool(cfg.eval.get("compute_fid", True))
    compute_fvd = bool(cfg.eval.get("compute_fvd", True))
    lpips_net = str(cfg.eval.get("lpips_net", "alex"))
    fid_stride = max(1, int(cfg.eval.get("fid_frame_stride", 1)))
    save_n = int(cfg.eval.get("save_n_videos", 8))
    fps = int(cfg.eval.get("fps", 8))
    out_dir = Path(cfg.eval.output_dir)

    fvd_handle = None
    fvd_backend = str(cfg.eval.fvd.get("backend", "i3d"))
    if compute_fvd:
        try:
            fvd_handle = fvd_mod.load_fvd_model(fvd_backend, cfg.eval.fvd.get("i3d_weights"), torch.device(device))
        except Exception as exc:
            logger.warning("FVD disabled (could not load backend '%s'): %r", fvd_backend, exc)
            fvd_handle = None

    lpips_ok = True
    rows = []
    fid_gen = fid_real = fvd_gen = fvd_real = None
    n_saved = 0

    for i, idx in enumerate(tqdm(my_indices, disable=not is_main, desc="eval")):
        sample = val_ds[int(idx)]
        video = sample["video"]                 # [3, T, H, W] in [-1, 1]
        action_gt = sample["action"]            # [Ta, 3] normalized
        proprio = sample["proprio"]             # [Ta, 4] normalized
        num_frames = int(video.shape[1])
        action_horizon = int(action_gt.shape[0])

        with torch.no_grad():
            pred = model.infer(
                prompt=None,
                input_image=video[:, 0].unsqueeze(0),     # [1,3,H,W] in [-1,1]; infer casts dtype/device
                num_frames=num_frames,
                action=None,                              # open-loop (video not action-conditioned)
                action_horizon=action_horizon,
                proprio=proprio[0],                       # anchor row -> [4]
                context=sample["context"],
                context_mask=sample["context_mask"],
                text_cfg_scale=float(cfg.eval.get("text_cfg_scale", 1.0)),
                num_inference_steps=int(cfg.eval.get("num_inference_steps", 30)),
                sigma_shift=cfg.eval.get("sigma_shift", None),
                seed=seed,
                tiled=tiled,
                test_action_with_infer_action=False,
            )
        pred_video = pred["video"]      # list[PIL]
        pred_action = pred["action"]    # [Ta, 3] normalized, cpu

        # ---- action metrics ----
        proprio_btd = proprio.unsqueeze(0)  # [1, Ta, 4]
        pred_denorm = _denorm_action(processor, pred_action, proprio_btd)
        gt_denorm = _denorm_action(processor, action_gt, proprio_btd)
        row = {"idx": int(idx)}
        row.update(open_loop_trajectory_metrics(pred_denorm[0], gt_denorm[0]))

        # ---- video tensors + per-frame metrics ----
        pred_vt = pil_frames_to_video_tensor(pred_video)  # [3,T,H,W] in [0,1]
        gt_vt = ((video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
        row["psnr_rg"] = video_psnr(pred_vt, gt_vt)
        row["ssim_rg"] = video_ssim(pred_vt, gt_vt)
        if lpips_ok:
            try:
                from fastwam.utils.perceptual_metrics import lpips_video
                row["lpips"] = lpips_video(pred_vt, gt_vt, net=lpips_net, device=torch.device(device))
            except Exception as exc:
                lpips_ok = False
                if is_main:
                    logger.warning("LPIPS disabled (%r); `pip install lpips` to enable.", exc)

        vae_vt = None
        if compute_vae:
            with torch.no_grad():
                lat = model._encode_video_latents(
                    video.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype), tiled=tiled
                )
                vae_frames = model._decode_latents(lat, tiled=tiled)
            vae_vt = pil_frames_to_video_tensor(vae_frames)
            row["psnr_dg"] = video_psnr(vae_vt, gt_vt)
            row["ssim_dg"] = video_ssim(vae_vt, gt_vt)
            row["psnr_rd"] = video_psnr(pred_vt, vae_vt)
            row["ssim_rd"] = video_ssim(pred_vt, vae_vt)
        rows.append(row)

        # ---- FID (frame-level) + FVD (clip-level) streaming stats ----
        if compute_fid:
            gen_frames = pred_vt[:, ::fid_stride].permute(1, 0, 2, 3)  # [Nf,3,H,W]
            real_frames = gt_vt[:, ::fid_stride].permute(1, 0, 2, 3)
            fid_gen = _accumulate(fid_gen, inception_features(gen_frames, device=torch.device(device)).numpy().astype(np.float64))
            fid_real = _accumulate(fid_real, inception_features(real_frames, device=torch.device(device)).numpy().astype(np.float64))
        if fvd_handle is not None:
            fvd_gen = _accumulate(fvd_gen, fvd_mod.fvd_features(pred_vt.unsqueeze(0), fvd_handle).numpy().astype(np.float64))
            fvd_real = _accumulate(fvd_real, fvd_mod.fvd_features(gt_vt.unsqueeze(0), fvd_handle).numpy().astype(np.float64))

        if is_main and n_saved < save_n:
            _save_sample(out_dir, int(idx), pred_vt, vae_vt, gt_vt, pred_denorm[0], gt_denorm[0], val_ds, fps)
            n_saved += 1

    # ---- gather across ranks ----
    payload = {"rows": rows, "fid_gen": fid_gen, "fid_real": fid_real, "fvd_gen": fvd_gen, "fvd_real": fvd_real}
    if world > 1 and dist.is_available() and dist.is_initialized():
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    if not is_main:
        return

    all_rows = [r for p in gathered for r in p["rows"]]
    fid_gen = _combine([p["fid_gen"] for p in gathered])
    fid_real = _combine([p["fid_real"] for p in gathered])
    fvd_gen = _combine([p["fvd_gen"] for p in gathered])
    fvd_real = _combine([p["fvd_real"] for p in gathered])

    video_summary = {k: _nanmean(all_rows, k) for k in VIDEO_SCALAR_KEYS}
    fvd_val = _frechet_from_accs(fvd_real, fvd_gen)
    fid_val = _frechet_from_accs(fid_real, fid_gen)
    if fvd_val is not None:
        video_summary["fvd"] = fvd_val
    if fid_val is not None:
        video_summary["fid"] = fid_val

    summary = {
        "checkpoint": str(cfg.ckpt),
        "version": str(cfg.data.val.get("version")),
        "split": str(cfg.data.val.get("split")),
        "num_samples_requested": int(num),
        "num_samples_evaluated": len(all_rows),
        "num_inference_steps": int(cfg.eval.get("num_inference_steps", 30)),
        "text_cfg_scale": float(cfg.eval.get("text_cfg_scale", 1.0)),
        "action_conditioning": "open_loop",
        "fvd_backend": fvd_backend if fvd_val is not None else None,
        "fid_frame_stride": fid_stride,
        "action": {k: _nanmean(all_rows, k) for k in ACTION_KEYS},
        "video": video_summary,
        "duration_sec": round(time.time() - t0, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    csv_keys = ["idx", "num_waypoints"] + ACTION_KEYS + VIDEO_SCALAR_KEYS
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_rows, key=lambda r: r["idx"]):
            w.writerow(r)

    logger.info("nuScenes eval done in %.1fs over %d samples.", summary["duration_sec"], len(all_rows))
    logger.info("action: %s", json.dumps(summary["action"]))
    logger.info("video:  %s", json.dumps(summary["video"]))
    logger.info("Wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
