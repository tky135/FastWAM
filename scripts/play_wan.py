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
        --num-frames 81 --num-steps 20 --width 1280 --height 704 --text-cfg 5.0

Notes:
  - The checkpoint is fetched on first run via FastWAM's loader (HuggingFace or ModelScope,
    depending on env). Subsequent runs hit the local cache.
  - `--num-frames` must satisfy `T % 4 == 1` (Wan VAE temporal patching); if it doesn't,
    we round it up. The model was trained for up to 121 frames (5 s @ 24 fps).
  - The conditioning image is locked to an exact `(--width, --height)` via scale-to-cover
    + center-crop. Both dims must be multiples of 32 (VAE 16× spatial compression × DiT
    2×2 patch). Defaults `--width 1280 --height 704` match the Wan2.2-TI2V-5B landscape
    training preset (`Wan2.2/wan/configs/__init__.py:46`). For a nuScenes 1600x900 frame,
    this scales 1600→1280 (width-locked), produces a 1280x720 intermediate, then center-
    crops 8 px top + 8 px bottom to land at 1280x704.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastwam.runtime import create_wan22_model
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.video_io import save_mp4

logger = get_logger(__name__)


def _ensure_wan22_on_path() -> None:
    """Make sure the official `wan` package can be imported. Idempotent."""
    try:
        import wan  # noqa: F401  # type: ignore[import-not-found]
        return
    except ImportError:
        pass

    candidates: list[Path] = []
    env_path = os.environ.get("WAN22_PATH")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve()
    for offset in (2, 1):
        try:
            candidates.append(here.parents[offset] / "Wan2.2")
        except IndexError:
            pass
    candidates.extend([
        Path.cwd() / "Wan2.2",
        Path.cwd().parent / "Wan2.2",
    ])

    for candidate in candidates:
        if (candidate / "wan" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            import wan  # noqa: F401  # type: ignore[import-not-found]
            return

    raise ImportError(
        "Cannot find the official Wan2.2 `wan` package. Set WAN22_PATH=/path/to/Wan2.2 or "
        "clone the Wan2.2 repo next to FastWAM (so its `wan/` directory is importable)."
    )


def _import_official_wan_vae():
    """Locate the official `Wan2_2_VAE` class from a Wan2.2 source tree.

    Used by the `--official-vae` bisect flag to swap FastWAM's `WanVideoVAE38` for the
    upstream implementation while keeping the rest of the FastWAM pipeline intact.
    """
    _ensure_wan22_on_path()
    from wan.modules.vae2_2 import Wan2_2_VAE  # type: ignore[import-not-found]
    return Wan2_2_VAE


def _import_official_t5_encoder():
    """Locate the official `T5EncoderModel` class from a Wan2.2 source tree.

    Used by the `--official-text-encoder` bisect flag to swap FastWAM's
    `WanTextEncoder` + `HuggingfaceTokenizer` pipeline for the upstream T5 wrapper.
    """
    _ensure_wan22_on_path()
    from wan.modules.t5 import T5EncoderModel  # type: ignore[import-not-found]
    return T5EncoderModel


def _import_official_wan_model():
    """Locate the official `WanModel` class from a Wan2.2 source tree.

    Used by the `--official-dit` bisect flag to swap FastWAM's `WanVideoDiT` for the
    upstream `WanModel` (from `wan.modules.model`). This is the biggest single bisect
    step — the forward signature is different, so an adapter does the input/output
    translation.
    """
    _ensure_wan22_on_path()
    from wan.modules.model import WanModel  # type: ignore[import-not-found]
    return WanModel


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


# Curated negative prompt that matches the official Wan2.2 inference defaults
# (Wan2.2/wan/configs/shared_config.py:19). The model was trained against this exact
# anti-recipe; using "" instead means CFG has no specific failure-mode to push away
# from, which on driving scenes manifests as "all dark" or saturated output at CFG≥3.
WAN22_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# Wan2.2-TI2V-5B DiT config. These dims are baked into the pretrained weights — don't
# change them unless you're loading a different checkpoint. Most fields mirror
# `configs/model/fastwam.yaml:video_dit_config`, but `video_attention_mask_mode` is
# explicitly set to "bidirectional" (the vanilla Wan2.2 pretraining setting and the
# WanVideoDiT class default at wan_video_dit.py:336). The FastWAM YAML uses
# "first_frame_causal" for its action-conditioned training, which is NOT what the
# pretrained Wan2.2 weights were trained under — running vanilla weights under that
# mask cripples the first-frame representation and produces garbage video.
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
    "video_attention_mask_mode": "bidirectional",
    "action_conditioned": False,
}


def _load_and_resize_image(path: str, target_w: int, target_h: int) -> torch.Tensor:
    """Load `path` and produce an exact `(target_w, target_h)` tensor via scale-to-cover +
    center-crop. Both target dims must be multiples of 32 (Wan2.2 VAE 16× × DiT 2×2 patch).

    For a 1600x900 nuScenes CAM_FRONT image with the defaults (1280x704):
      scale = max(1280/1600, 704/900) = max(0.8, 0.7822) = 0.8
      intermediate = 1280 x 720, then center-crop removes 8 px top + 8 px bottom.
    """
    if target_w % 32 != 0 or target_h % 32 != 0:
        raise ValueError(
            f"--width and --height must both be multiples of 32 (VAE 16× × DiT patch 2), "
            f"got width={target_w}, height={target_h}."
        )
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    img = img.resize((resized_w, resized_h), Image.LANCZOS)
    # Center-crop. Either resized_w==target_w (width-bound) or resized_h==target_h
    # (height-bound) by construction; the other dim is >= the target.
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    arr = np.asarray(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 127.5 - 1.0
    return tensor.unsqueeze(0)  # [1, 3, target_h, target_w]


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
        "--text-cfg", type=float, default=5.0,
        help="Text classifier-free guidance scale. Default 5.0 matches Wan2.2 official `sample_guide_scale` for TI2V-5B (configs/wan_ti2v_5B.py:35). Set to 1.0 to disable CFG and skip the unconditional pass entirely.",
    )
    parser.add_argument(
        "--negative-prompt", default=WAN22_DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt used when text_cfg != 1.0. Default is the official Wan2.2 curated anti-recipe (Chinese). Pass an empty string to disable, or your own string to override.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Target output width in pixels (must be a multiple of 32 — VAE 16× × DiT patch 2). Default 1280 matches Wan2.2-TI2V-5B landscape training preset.",
    )
    parser.add_argument(
        "--height", type=int, default=704,
        help="Target output height in pixels (must be a multiple of 32). Default 704 matches Wan2.2-TI2V-5B landscape training preset.",
    )
    parser.add_argument("--fps", type=int, default=8, help="Output video fps.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiled", action="store_true", help="Tiled VAE encode/decode (slower, less memory).")
    parser.add_argument("--save-frames", action="store_true", help="Also dump each frame as PNG next to the MP4.")
    parser.add_argument("--sigma-shift", type=float, default=None, help="Override sigma shift for inference schedule.")
    parser.add_argument(
        "--sample-solver", choices=["euler", "unipc"], default="unipc",
        help=(
            "ODE solver for the flow-matching denoising loop. 'unipc' (default) matches the "
            "official Wan2.2 inference at `wan/textimage2video.py:527-534` — multistep "
            "predictor-corrector, much sharper at <50 steps. 'euler' is FastWAM's flat Euler "
            "integrator, kept for backwards compatibility. Requires diffusers installed for 'unipc'."
        ),
    )
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--tokenizer-max-len", type=int, default=512)
    parser.add_argument(
        "--local-dir", default=None,
        help=(
            "Path to a single official Wan2.2-TI2V-5B checkpoint directory. When set, the "
            "DiT, VAE (fp32), T5, and tokenizer are all loaded directly from this directory "
            "— no DiffSynth-Studio redirect, no download. This is the recommended way to run "
            "against the official weights; --vae-pth/--vae-fp32 become unnecessary."
        ),
    )
    parser.add_argument(
        "--vae-pth", default=None,
        help=(
            "Override path to the official Wan2.2_VAE.pth (or any fp32 weight file). "
            "When set, bypasses the DiffSynth-Studio bf16-converted mirror that FastWAM "
            "redirects to by default. Recommended for inference quality — the default mirror "
            "is bf16-quantized, which visibly degrades VAE decoding when paired with bf16 DiT inference."
        ),
    )
    parser.add_argument(
        "--vae-fp32", action="store_true",
        help="Run the VAE in fp32 regardless of --dtype. Implicitly enabled when --vae-pth is set. Mirrors official `Wan2_2_VAE(dtype=torch.float)` behavior.",
    )
    parser.add_argument(
        "--official-vae", action="store_true",
        help=(
            "BISECT: replace FastWAM's `WanVideoVAE38` with the official `Wan2_2_VAE` class "
            "from Wan2.2/wan/modules/vae2_2.py. Requires --vae-pth pointing at the official "
            ".pth weights. Useful for narrowing down whether the FastWAM VAE *class* (not "
            "weights) has an implementation difference from the upstream pipeline."
        ),
    )
    parser.add_argument(
        "--official-text-encoder", action="store_true",
        help=(
            "BISECT: replace FastWAM's `WanTextEncoder` + `HuggingfaceTokenizer` with the "
            "official `T5EncoderModel` (Wan2.2/wan/modules/t5.py). Useful for narrowing "
            "down whether the FastWAM text-encoding pipeline diverges from upstream. "
            "Requires --t5-pth and --t5-tokenizer (or auto-derived from --vae-pth's dir)."
        ),
    )
    parser.add_argument(
        "--t5-pth", default=None,
        help="Path to official `models_t5_umt5-xxl-enc-bf16.pth`. If unset and --official-text-encoder is on, auto-derived from --vae-pth's parent directory.",
    )
    parser.add_argument(
        "--t5-tokenizer", default=None,
        help="Path to official `google/umt5-xxl` tokenizer directory. If unset and --official-text-encoder is on, auto-derived from --vae-pth's parent directory.",
    )
    parser.add_argument(
        "--official-dit", action="store_true",
        help=(
            "BISECT: replace FastWAM's `WanVideoDiT` with the official `WanModel` "
            "(Wan2.2/wan/modules/model.py). Different forward signature, so an adapter "
            "translates inputs/outputs. Requires --dit-dir or --vae-pth (the safetensors "
            "shards live in the same directory)."
        ),
    )
    parser.add_argument(
        "--dit-dir", default=None,
        help="Path to the directory containing `config.json` + `diffusion_pytorch_model-*.safetensors` for the official `WanModel.from_pretrained`. If unset and --official-dit is on, auto-derived from --vae-pth's parent directory.",
    )
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
    # In --local-dir mode the loader sources the VAE from the dir's fp32 Wan2.2_VAE.pth and
    # defaults it to fp32 automatically, so no separate vae override is needed.
    vae_dtype_override = torch.float32 if (args.vae_fp32 or args.vae_pth) else None
    if args.local_dir is not None:
        logger.info("Loading all components directly from --local-dir=%s (no DiffSynth redirect).", args.local_dir)
    elif vae_dtype_override is not None:
        logger.info(
            "Loading VAE at fp32 (override). %s",
            f"Source: {args.vae_pth}" if args.vae_pth else "Source: default mirror (will be cast bf16→fp32 on load).",
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
        vae_path=args.vae_pth,
        vae_dtype=vae_dtype_override,
        local_dir=args.local_dir,
    )
    logger.info("Loaded. Local checkpoint paths:")
    for name, path in getattr(model, "model_paths", {}).items():
        logger.info(f"  {name}: {path}")

    if args.official_vae:
        if not args.vae_pth:
            raise SystemExit("--official-vae requires --vae-pth (path to the official Wan2.2_VAE.pth).")
        Wan2_2_VAE_cls = _import_official_wan_vae()
        logger.info(
            "BISECT: swapping FastWAM `WanVideoVAE38` for official `Wan2_2_VAE` "
            f"(weights={args.vae_pth})."
        )
        # Instantiate the official VAE (its own constructor handles dtype=fp32 + device move
        # and loads the .pth via `torch.load`).
        official_vae = Wan2_2_VAE_cls(vae_pth=args.vae_pth, device=args.device)

        # Monkey-patch the two methods on Wan22Core that touch the VAE during inference.
        # `_encode_video_latents` is training-only and isn't called from `infer`, so we
        # leave it alone. The official VAE expects list-wrapped `[3, T, H, W]` tensors and
        # returns list-wrapped outputs — we adapt the in/out shape to FastWAM's contract.
        import types

        def _encode_via_official(self, input_image, tiled=False, **kwargs):  # noqa: ARG001
            if input_image.ndim == 3:
                input_image = input_image.unsqueeze(0)
            if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,H,W], got {tuple(input_image.shape)}"
                )
            # [1,3,H,W] → [3,1,H,W] in fp32 (official VAE expects this).
            image = input_image.to(device=self.device, dtype=torch.float32)[0].unsqueeze(1)
            z_list = official_vae.encode([image])
            # z_list[0]: [z_dim, T_lat, H_lat, W_lat]. Add batch dim to match FastWAM contract.
            return z_list[0].unsqueeze(0)

        def _decode_via_official(self, latents, tiled=False, **kwargs):  # noqa: ARG001
            # latents: [1, z_dim, T_lat, H_lat, W_lat]; official expects [z_dim, T_lat, H_lat, W_lat] in a list.
            latent = latents.squeeze(0).to(dtype=torch.float32)
            video = official_vae.decode([latent])[0]  # [3, T_pixel, H_pixel, W_pixel] in [-1, 1]
            video = video.detach().float().clamp(-1, 1)
            video = ((video + 1.0) * 127.5).to(torch.uint8).cpu()
            frames = []
            for t in range(video.shape[1]):
                frame = video[:, t].permute(1, 2, 0).numpy()
                frames.append(Image.fromarray(frame))
            return frames

        model._encode_input_image_latents_tensor = types.MethodType(_encode_via_official, model)
        model._decode_latents = types.MethodType(_decode_via_official, model)
        # The remaining attributes Wan22Core reads off `self.vae` (`temporal_downsample_factor`,
        # `upsampling_factor`, `model.z_dim`) live on the FastWAM VAE wrapper, so we leave
        # `model.vae` itself untouched — only the encode/decode pathways are swapped. The
        # vae_dtype attribute is read in the two methods we just replaced; set it for clarity.
        model.vae_dtype = torch.float32

    if args.official_text_encoder:
        t5_pth = args.t5_pth
        t5_tokenizer = args.t5_tokenizer
        if (t5_pth is None or t5_tokenizer is None):
            if not args.vae_pth:
                raise SystemExit(
                    "--official-text-encoder needs --t5-pth and --t5-tokenizer, or --vae-pth "
                    "so they can be auto-derived from the same checkpoint directory."
                )
            ckpt_dir = Path(args.vae_pth).parent
            if t5_pth is None:
                t5_pth = str(ckpt_dir / "models_t5_umt5-xxl-enc-bf16.pth")
            if t5_tokenizer is None:
                t5_tokenizer = str(ckpt_dir / "google" / "umt5-xxl")
        if not Path(t5_pth).is_file():
            raise SystemExit(f"--t5-pth not found: {t5_pth}")
        if not Path(t5_tokenizer).is_dir():
            raise SystemExit(f"--t5-tokenizer not found: {t5_tokenizer}")

        T5EncoderModel_cls = _import_official_t5_encoder()
        logger.info(
            "BISECT: swapping FastWAM `WanTextEncoder`+`HuggingfaceTokenizer` for official "
            f"`T5EncoderModel` (checkpoint={t5_pth}, tokenizer={t5_tokenizer})."
        )
        # The official wrapper bundles tokenizer + encoder and exposes `__call__(texts, device)`.
        # `text_len=512` matches FastWAM's tokenizer_max_len default and the Wan2.2 config.
        # `dtype=torch.bfloat16` matches the official `WanTI2V.__init__` default for TI2V-5B.
        official_text_encoder = T5EncoderModel_cls(
            text_len=args.tokenizer_max_len,
            dtype=model_dtype,
            device=args.device,
            checkpoint_path=t5_pth,
            tokenizer_path=t5_tokenizer,
        )

        text_len = int(args.tokenizer_max_len)

        def _encode_prompt_via_official(self, prompt):  # noqa: ARG001
            # Official returns a list of variable-length tensors truncated to actual seq_len.
            # FastWAM's downstream expects [B, text_len, D] context + [B, text_len] bool mask;
            # mirror official `WanModel.forward` (model.py:472-478) which pads each truncated
            # context to `text_len` with zeros, then build the matching mask.
            if isinstance(prompt, str):
                texts = [prompt]
            else:
                texts = list(prompt)
            ctx_list = official_text_encoder(texts, self.device)
            padded_list = []
            mask_rows = []
            for u in ctx_list:
                actual = u.shape[0]
                if actual < text_len:
                    pad = u.new_zeros(text_len - actual, u.shape[1])
                    u_padded = torch.cat([u, pad], dim=0)
                else:
                    u_padded = u[:text_len]
                padded_list.append(u_padded)
                row = torch.zeros(text_len, dtype=torch.bool, device=u.device)
                row[: min(actual, text_len)] = True
                mask_rows.append(row)
            context = torch.stack(padded_list, dim=0).to(self.device)  # [B, text_len, D]
            mask = torch.stack(mask_rows, dim=0).to(self.device)         # [B, text_len]
            return context, mask

        import types  # ensure imported (also imported in the VAE branch)
        model.encode_prompt = types.MethodType(_encode_prompt_via_official, model)

    if args.official_dit:
        dit_dir = args.dit_dir
        if dit_dir is None:
            if not args.vae_pth:
                raise SystemExit(
                    "--official-dit needs --dit-dir, or --vae-pth so the directory can be "
                    "auto-derived from the same checkpoint layout."
                )
            dit_dir = str(Path(args.vae_pth).parent)
        if not (Path(dit_dir) / "config.json").is_file():
            raise SystemExit(
                f"--dit-dir does not look like a Wan2.2-TI2V-5B checkpoint dir (no config.json at {dit_dir})."
            )

        WanModel_cls = _import_official_wan_model()
        logger.info(
            "BISECT: swapping FastWAM `WanVideoDiT` for official `WanModel` "
            f"(checkpoint_dir={dit_dir}, dtype={args.dtype})."
        )
        # `from_pretrained` loads `config.json` + shard the safetensors. We cast to the same
        # dtype FastWAM was using (model_dtype, typically bf16) so the autocast policy stays
        # consistent.
        official_dit = (
            WanModel_cls.from_pretrained(dit_dir)
            .eval()
            .requires_grad_(False)
            .to(device=args.device, dtype=model_dtype)
        )

        def _model_fn_via_official(self, latents, timestep, context, context_mask=None,
                                   action=None, fuse_vae_embedding_in_latents=False):  # noqa: ARG001
            # Translate FastWAM's `_model_fn` input shape to the official WanModel.forward
            # signature (see `Wan2.2/wan/modules/model.py:410-497` and the per-token timestep
            # construction in `wan/textimage2video.py:573-578`).
            if latents.shape[0] != 1:
                raise NotImplementedError("Official `WanModel` adapter currently only supports B=1.")
            T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
            tokens_per_frame = (H_lat // 2) * (W_lat // 2)  # patch_size=(1,2,2)
            seq_len = T_lat * tokens_per_frame

            # Build per-token timestep: 0 for first-frame tokens (the conditioning anchor),
            # `timestep` for every other position. Mirrors the masked construction at
            # textimage2video.py:573-578 (`mask2[0][0][:, ::2, ::2] * timestep`, where mask2 is
            # zero on the first temporal index — see `wan/utils/utils.py:masks_like`).
            t_scalar = timestep.reshape(-1)[0]
            per_token_t = t_scalar.expand(seq_len).clone()
            per_token_t[:tokens_per_frame] = 0
            per_token_t = per_token_t.unsqueeze(0)  # [1, seq_len]

            # Convert FastWAM's padded context back to the official's variable-length list.
            if context_mask is not None:
                actual_len = int(context_mask[0].sum().item())
            else:
                actual_len = context.shape[1]
            ctx_truncated = context[0, :actual_len].contiguous()  # [actual_len, D]

            # WanModel.forward expects List[Tensor] for x and context, no batch dim per item.
            latent_input = [latents[0]]
            out_list = official_dit(latent_input, t=per_token_t, context=[ctx_truncated], seq_len=seq_len)
            # out_list: List[Tensor] each [C_out, T_lat, H_lat, W_lat] (post-unpatchify).
            return out_list[0].unsqueeze(0)  # add batch dim back: [1, C_out, T_lat, H_lat, W_lat]

        import types
        model._model_fn = types.MethodType(_model_fn_via_official, model)

    logger.info(f"Loading conditioning image: {args.image} (target {args.width}x{args.height}, scale-to-cover + center-crop).")
    input_image = _load_and_resize_image(args.image, args.width, args.height)
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
        solver=args.sample_solver,
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
