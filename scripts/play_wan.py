"""Play with the pretrained Wan2.2-TI2V-5B model via FastWAM's loader.

Loads the vanilla image-to-video Wan2.2 checkpoint (no FastWAM action head, no MoT) and
runs the standard I2V denoising loop on a user-supplied (image, prompt) pair. Saves the
generated video as an MP4 plus, optionally, per-frame PNGs.

Defaults are tuned to match FastWAM's nuScenes-task eval pipeline:
  - 9 frames at 224x384 resolution (matches `configs/data/nuscenes_1cam224.yaml`)
  - 10 denoising steps, text_cfg_scale=1.0 (matches `trainer.py:402-414` + `train.yaml:24`)
  - prompt = nuScenes templated default (matches `RobotVideoDataset._get`)

Example (nuScenes-eval generation, all defaults):
    python scripts/play_wan.py --image path/to/cam_front.jpg \\
        --out runs/play_wan/nuscenes.mp4

Example (higher-quality I2V exploration):
    python scripts/play_wan.py --image path/to/scene.jpg \\
        --prompt "A serene mountain landscape at sunset, gentle camera dolly forward." \\
        --num-frames 81 --num-steps 20 --max-side 704 --text-cfg 5.0

Notes:
  - The checkpoint is fetched on first run via FastWAM's loader (HuggingFace or ModelScope,
    depending on env). Subsequent runs hit the local cache.
  - `--num-frames` must satisfy `T % 4 == 1` (Wan VAE temporal patching); if it doesn't,
    we round it up. The model was trained for up to 121 frames (5 s @ 24 fps).
  - The conditioning image is rescaled so its long side equals `--max-side`, then both
    dims are rounded up to multiples of 16. For a nuScenes 1600x900 frame at
    `--max-side 384`, this lands at 384x224 — the training resolution.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastwam.runtime import create_wan22_model
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.video_io import save_mp4

logger = get_logger(__name__)


# nuScenes-eval defaults. Sourced from:
#   - configs/data/nuscenes_1cam224.yaml: num_frames=9, video_size=[224, 384],
#     override_instruction="Front-view driving scene from a moving vehicle."
#   - src/fastwam/datasets/lerobot/robot_video_dataset.py:DEFAULT_PROMPT — the templating
#     wrapper applied to override_instruction before T5 encoding.
#   - configs/train.yaml: eval_num_inference_steps=10
#   - src/fastwam/trainer.py:402-414: eval calls model.infer with text_cfg_scale=1.0,
#     num_inference_steps=eval_num_inference_steps, seed=42, tiled=False.
NUSCENES_DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: "
    "Front-view driving scene from a moving vehicle."
)


# Wan2.2-TI2V-5B DiT config. These dims are baked into the pretrained weights — don't
# change them unless you're loading a different checkpoint. Mirrors
# `configs/model/fastwam.yaml:video_dit_config` minus the action/MoT-only fields.
DEFAULT_DIT_CONFIG = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "hidden_dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "attn_head_dim": 128,
    "num_layers": 30,
    "eps": 1.0e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
    "use_gradient_checkpointing": False,
    "video_attention_mask_mode": "first_frame_causal",
    "action_conditioned": False,
}


def _round_up_to_multiple(x: int, m: int) -> int:
    return max(m, ((x + m - 1) // m) * m)


def _load_and_resize_image(path: str, max_side: int) -> torch.Tensor:
    """Load `path`, rescale long side to `max_side` (preserving aspect), round both
    dimensions up to multiples of 16, and return a [1, 3, H, W] tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = float(max_side) / float(max(h, w))
    new_h = _round_up_to_multiple(int(round(h * scale)), 16)
    new_w = _round_up_to_multiple(int(round(w * scale)), 16)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 127.5 - 1.0
    return tensor.unsqueeze(0)  # [1, 3, H, W]


def main():
    parser = argparse.ArgumentParser(
        description="Run pretrained Wan2.2-TI2V-5B via FastWAM's loader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to the conditioning first-frame image.")
    parser.add_argument(
        "--prompt", default=NUSCENES_DEFAULT_PROMPT,
        help="Text prompt. Default mirrors the nuScenes-task eval prompt (the templated `DEFAULT_PROMPT.format(task=override_instruction)` string).",
    )
    parser.add_argument("--out", default="runs/play_wan/out.mp4", help="Output MP4 path.")
    parser.add_argument(
        "--num-frames", type=int, default=9,
        help="Total frames including the conditioning first frame; must satisfy T %% 4 == 1. Default 9 matches nuScenes data config (T_video=9).",
    )
    parser.add_argument(
        "--num-steps", type=int, default=10,
        help="Denoising steps. Default 10 matches `configs/train.yaml:eval_num_inference_steps`.",
    )
    parser.add_argument(
        "--text-cfg", type=float, default=1.0,
        help="Text classifier-free guidance scale. Default 1.0 matches trainer eval (no CFG).",
    )
    parser.add_argument(
        "--negative-prompt", default=None,
        help="Negative prompt used when text_cfg != 1.0. Empty string by default inside the model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-side", type=int, default=384,
        help="Long-side resolution after rescale (rounded to multiple of 16). Default 384 → ~224x384 for nuScenes 16:9 aspect, matching training video_size.",
    )
    parser.add_argument("--fps", type=int, default=8, help="Output video fps.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiled", action="store_true", help="Tiled VAE encode/decode (slower, less memory).")
    parser.add_argument("--save-frames", action="store_true", help="Also dump each frame as PNG next to the MP4.")
    parser.add_argument("--sigma-shift", type=float, default=None, help="Override sigma shift for inference schedule.")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--tokenizer-max-len", type=int, default=512)
    args = parser.parse_args()

    setup_logging()

    if args.num_frames % 4 != 1:
        rounded = ((args.num_frames - 1) // 4) * 4 + 1
        logger.warning(
            f"--num-frames={args.num_frames} doesn't satisfy T %% 4 == 1; using {rounded} instead."
        )
        args.num_frames = rounded

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU. This will be extremely slow.")
        args.device = "cpu"
        if args.dtype != "fp32":
            logger.warning(f"Overriding --dtype to fp32 on CPU (was {args.dtype}).")
            model_dtype = torch.float32

    logger.info(
        f"Loading Wan2.2 components: model={args.model_id}, tokenizer={args.tokenizer_model_id}, "
        f"dtype={args.dtype}, device={args.device}."
    )
    model = create_wan22_model(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        dit_config=DEFAULT_DIT_CONFIG,
        tokenizer_max_len=args.tokenizer_max_len,
        train_shift=5.0,
        infer_shift=5.0,
        num_train_timesteps=1000,
        redirect_common_files=True,
        model_dtype=model_dtype,
        device=args.device,
    )
    logger.info("Loaded. Local checkpoint paths:")
    for name, path in getattr(model, "model_paths", {}).items():
        logger.info(f"  {name}: {path}")

    logger.info(f"Loading conditioning image: {args.image} (rescale long-side → {args.max_side}).")
    input_image = _load_and_resize_image(args.image, args.max_side)
    height, width = input_image.shape[-2:]
    logger.info(
        f"Conditioning frame {height}x{width}; generating {args.num_frames} frames over "
        f"{args.num_steps} denoising steps (text_cfg={args.text_cfg}, seed={args.seed})."
    )

    out = model.infer(
        prompt=args.prompt,
        input_image=input_image,
        num_frames=args.num_frames,
        text_cfg_scale=args.text_cfg,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        tiled=args.tiled,
    )
    frames = out["video"]
    logger.info(f"Generated {len(frames)} frames; writing outputs.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_mp4(frames, str(out_path), fps=args.fps)
    logger.info(f"MP4 → {out_path}")

    # Always save the conditioning frame next to the MP4 for visual reference. The
    # conditioning frame is index 0 of the generated sequence (Wan I2V anchors there).
    cond_path = out_path.with_name(out_path.stem + "_cond.png")
    frames[0].save(cond_path)
    logger.info(f"Conditioning frame → {cond_path}")

    if args.save_frames:
        frames_dir = out_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(frames_dir / f"frame_{i:04d}.png")
        logger.info(f"Per-frame PNGs → {frames_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
