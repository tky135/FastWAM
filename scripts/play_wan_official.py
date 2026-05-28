"""Run the official Wan2.2 TI2V-5B inference with the same CLI as play_wan.py.

Thin wrapper around `wan.WanTI2V.generate()` (the upstream pipeline at
third_party/Wan2.2/). Lets you A/B the same conditioning image + prompt + seed
against play_wan.py and see whether FastWAM's wrapper is the bottleneck.

Bypasses the `fastwam` package entirely — imports directly from third_party/Wan2.2.

You need the official Wan2.2-TI2V-5B checkpoint laid out as the upstream loader
expects (single directory containing `models_t5_umt5-xxl-enc-bf16.pth`,
`Wan2.2_VAE.pth`, `google/umt5-xxl/`, and the DiT weights). Pass its path via
`--ckpt-dir`. The simplest way to get it:

    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir <ckpt-dir>
    # or via ModelScope:
    modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir <ckpt-dir>

Example (matches play_wan.py's nuScenes defaults):
    python scripts/play_wan_official.py \\
        --image path/to/cam_front.jpg \\
        --ckpt-dir /path/to/Wan2.2-TI2V-5B \\
        --out runs/play_wan_official/out.mp4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image


# ---- locate and import the official `wan` package -----------------------------------
THIS_FILE = Path(__file__).resolve()
# scripts/ → FastWAM/ → third_party/ → Wan2.2/
WAN22_ROOT = THIS_FILE.parent.parent.parent / "Wan2.2"
if not WAN22_ROOT.is_dir():
    raise FileNotFoundError(
        f"Expected the official Wan2.2 repo at {WAN22_ROOT}. "
        "Clone https://github.com/Wan-Video/Wan2.2 there, or edit WAN22_ROOT in this script."
    )
sys.path.insert(0, str(WAN22_ROOT))

import wan  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


# Default prompt mirrors play_wan.py so A/B comparisons share the input.
NUSCENES_DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: "
    "Front-view driving scene from a moving vehicle."
)


def _compute_max_area(max_side: int, aspect_w_over_h: float = 16.0 / 9.0) -> int:
    """Convert play_wan.py-style `--max-side` into Wan2.2's `max_area` knob.

    Wan2.2's pipeline expects `max_area` (total pixel count) and infers the actual
    output H/W from the input image's aspect ratio + VAE/DiT stride multiples. To
    match play_wan.py's intent ("long side ≈ max_side"), we approximate
    `max_area = max_side * (max_side / aspect)` assuming a landscape input.
    """
    long_side = float(max_side)
    short_side = long_side / aspect_w_over_h
    return int(long_side * short_side)


def main():
    parser = argparse.ArgumentParser(
        description="Run official Wan2.2 TI2V-5B inference (parallel CLI to play_wan.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to the conditioning first-frame image.")
    parser.add_argument(
        "--prompt", default=NUSCENES_DEFAULT_PROMPT,
        help="Text prompt. Default mirrors play_wan.py's nuScenes prompt.",
    )
    parser.add_argument(
        "--negative-prompt", default="",
        help="Negative prompt. Empty string → official pipeline falls back to its built-in `sample_neg_prompt` (curated Chinese anti-recipe).",
    )
    parser.add_argument("--out", default="runs/play_wan_official/out.mp4", help="Output MP4 path.")
    parser.add_argument(
        "--num-frames", type=int, default=49,
        help="Total frames; must satisfy T %% 4 == 1. Rounded up if it doesn't.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=50,
        help="Denoising steps. Default 50 matches `ti2v_5B.sample_steps`.",
    )
    parser.add_argument(
        "--text-cfg", type=float, default=5.0,
        help="Classifier-free guidance scale. Default 5.0 matches `ti2v_5B.sample_guide_scale`.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use -1 for non-deterministic.")
    parser.add_argument(
        "--max-side", type=int, default=1280,
        help="Long-side resolution target. Converted to `max_area = max_side * (max_side / aspect)` for landscape input. Default 1280 ≈ official `1280*704`.",
    )
    parser.add_argument(
        "--aspect", type=float, default=16.0 / 9.0,
        help="Output aspect ratio (w/h) used to compute max_area from --max-side. Default 16/9 matches nuScenes / 1280*704.",
    )
    parser.add_argument("--fps", type=int, default=24, help="Output MP4 fps. Default 24 matches `ti2v_5B.sample_fps`.")
    parser.add_argument(
        "--sigma-shift", type=float, default=5.0,
        help="Sample shift for the flow scheduler. Default 5.0 matches `ti2v_5B.sample_shift`. For 480p generations the docs suggest 3.0.",
    )
    parser.add_argument(
        "--sample-solver", choices=["unipc", "dpm++"], default="unipc",
        help="ODE solver. Official default is unipc.",
    )
    parser.add_argument(
        "--ckpt-dir", required=True,
        help="Path to the official Wan2.2-TI2V-5B checkpoint directory.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload-model", action="store_true",
                        help="Offload subnets to CPU between calls (saves VRAM, slower).")
    parser.add_argument("--t5-cpu", action="store_true", help="Run T5 text encoder on CPU.")
    parser.add_argument(
        "--convert-model-dtype", action="store_true",
        help="Cast model weights to the config's `param_dtype` (bf16 by default for TI2V-5B).",
    )
    parser.add_argument("--save-frames", action="store_true",
                        help="Also dump per-frame PNGs next to the MP4.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("play_wan_official")

    if args.num_frames % 4 != 1:
        rounded = ((args.num_frames - 1) // 4) * 4 + 1
        logger.warning(f"--num-frames={args.num_frames} doesn't satisfy T %% 4 == 1; using {rounded}.")
        args.num_frames = rounded

    if not Path(args.ckpt_dir).is_dir():
        raise FileNotFoundError(
            f"--ckpt-dir not found: {args.ckpt_dir}. "
            "Download the official Wan2.2-TI2V-5B checkpoint there first (see this script's docstring)."
        )

    task_name = "ti2v-5B"
    cfg = WAN_CONFIGS[task_name]

    max_area = _compute_max_area(args.max_side, args.aspect)
    # `size` is largely vestigial in the TI2V preprocess path (it uses max_area + the
    # input image's aspect ratio), but the pipeline expects a tuple — pass an officially
    # supported preset matching the orientation implied by `--aspect`.
    size = SIZE_CONFIGS["1280*704"] if args.aspect >= 1.0 else SIZE_CONFIGS["704*1280"]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; this script will be unusably slow on CPU. Aborting.")
        return 1
    device_id = 0 if args.device == "cuda" else int(args.device.replace("cuda:", ""))

    logger.info(f"Loading official WanTI2V pipeline from {args.ckpt_dir}.")
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    logger.info(f"Loading conditioning image: {args.image}")
    img = Image.open(args.image).convert("RGB")
    logger.info(
        f"Input image {img.height}x{img.width}; max_area={max_area} (~{int(max_area ** 0.5)}^2 px)."
    )

    logger.info(
        f"Generating {args.num_frames} frames, {args.num_steps} steps, "
        f"guide={args.text_cfg}, shift={args.sigma_shift}, solver={args.sample_solver}, seed={args.seed}."
    )
    video = wan_ti2v.generate(
        args.prompt,
        img=img,
        size=size,
        max_area=max_area,
        frame_num=args.num_frames,
        shift=args.sigma_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.num_steps,
        guide_scale=args.text_cfg,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=args.offload_model,
    )
    if video is None:
        # Non-rank-0 in a distributed run — but we set rank=0, so this shouldn't happen.
        logger.error("WanTI2V.generate returned None (non-rank-0?). Aborting.")
        return 1
    logger.info(f"Generated tensor shape: {tuple(video.shape)}.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # save_video expects [B, C, T, H, W]; generate() returns (C, T, H, W).
    save_video(
        tensor=video[None].cpu(),
        save_file=str(out_path),
        fps=args.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info(f"MP4 → {out_path}")

    # Conditioning frame for visual reference (matches play_wan.py's convention).
    cond_path = out_path.with_name(out_path.stem + "_cond.png")
    cond = ((video[:, 0].cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    Image.fromarray(cond.permute(1, 2, 0).numpy()).save(cond_path)
    logger.info(f"Conditioning frame → {cond_path}")

    if args.save_frames:
        frames_dir = out_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        tensor = ((video.cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        for i in range(tensor.shape[1]):
            frame_np = tensor[:, i].permute(1, 2, 0).numpy()
            Image.fromarray(frame_np).save(frames_dir / f"frame_{i:04d}.png")
        logger.info(f"Per-frame PNGs → {frames_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
