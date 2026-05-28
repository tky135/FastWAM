"""Run FastWAM's `WanVideoDiT` and the official `WanModel` on the same input and
find the first block where their outputs diverge.

Use this AFTER `verify_dit_state_dict.py` confirms the state-dict load is clean.
The DiT classes share weights but the forward implementations differ — this script
localizes WHICH block (or sub-component) first introduces the divergence.

Methodology:
  1. Build a fixed random input: latent [1, 48, T_lat, H_lat, W_lat], scalar timestep,
     dummy context [1, 512, 4096]. Same seed for reproducibility.
  2. Load both models from the same safetensors shards.
  3. Register forward hooks on each model's `blocks.0` ... `blocks.{num_layers-1}`,
     capturing the input `x` tensor and output tensor at each block.
  4. Run a single forward pass through each.
  5. Compare:
       - Final output diff (sanity).
       - Per-block input and output diff (`abs().max()`).
  6. Report the first block where the OUTPUT diff first explodes above a threshold.

Usage:

    python scripts/diff_dit_forward.py --ckpt-dir /path/to/Wan-AI-Wan2.2-TI2V-5B-dir

If `--ckpt-dir` is omitted, the script will try to auto-resolve via FastWAM's loader.
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional


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


def _ensure_wan22_on_path() -> None:
    try:
        import wan  # noqa: F401
        return
    except ImportError:
        pass
    import os
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
    candidates.extend([Path.cwd() / "Wan2.2", Path.cwd().parent / "Wan2.2"])
    for c in candidates:
        if (c / "wan" / "__init__.py").is_file():
            sys.path.insert(0, str(c))
            import wan  # noqa: F401
            return
    raise ImportError("Cannot find the official Wan2.2 `wan` package. Set WAN22_PATH=...")


def _resolve_ckpt_dir(explicit: Optional[str], model_id: str) -> str:
    if explicit:
        return explicit
    # Find via FastWAM's loader — it caches the model under a known path.
    from fastwam.models.wan22.helpers.io import ModelConfig
    cfg = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    cfg.download_if_necessary()
    path = cfg.path
    if isinstance(path, list):
        path = path[0]
    # cfg.path is a shard file; its parent dir is the checkpoint dir.
    return str(Path(path).parent)


def main():
    parser = argparse.ArgumentParser(
        description="Diff FastWAM WanVideoDiT vs official WanModel block-by-block.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-dir", default=None,
                        help="Directory containing config.json + diffusion_pytorch_model-*.safetensors. "
                             "If unset, resolved via FastWAM's ModelConfig.")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--timestep", type=float, default=500.0,
                        help="Scalar diffusion timestep to feed to both models.")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="abs().max() over the block output that counts as 'diverged'.")
    parser.add_argument("--show-each-block", action="store_true",
                        help="Print the per-block input/output diff for every block.")
    args = parser.parse_args()

    import torch  # heavy — defer
    from fastwam.models.wan22.helpers.io import load_state_dict
    from fastwam.models.wan22.helpers.loader import _validate_dit_config
    from fastwam.models.wan22.helpers.state_dict_converters import (
        wan_video_dit_state_dict_converter,
    )
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT

    _ensure_wan22_on_path()
    from wan.modules.model import WanModel  # type: ignore[import-not-found]

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    ckpt_dir = _resolve_ckpt_dir(args.ckpt_dir, args.model_id)
    print(f"Checkpoint dir: {ckpt_dir}")

    # ---- 1. Load both models from the same weights -------------------------------------
    print("\nLoading FastWAM WanVideoDiT...")
    validated = _validate_dit_config(DEFAULT_DIT_CONFIG)
    fw_model = WanVideoDiT(**validated)
    shards = sorted(glob.glob(str(Path(ckpt_dir) / "diffusion_pytorch_model*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards in {ckpt_dir}")
    fw_state = load_state_dict(shards if len(shards) > 1 else shards[0],
                               torch_dtype=torch_dtype, device="cpu")
    fw_state = wan_video_dit_state_dict_converter(fw_state)
    inc = fw_model.load_state_dict(fw_state, strict=False)
    if inc.missing_keys or inc.unexpected_keys:
        print(f"  WARNING: FastWAM load is incomplete. missing={len(inc.missing_keys)}, "
              f"unexpected={len(inc.unexpected_keys)}.")
    fw_model = fw_model.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

    print("Loading official WanModel...")
    of_model = WanModel.from_pretrained(ckpt_dir).to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

    # ---- 2. Build a fixed input ---------------------------------------------------------
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    T_lat = (args.num_frames - 1) // 4 + 1
    H_lat = args.height // 16
    W_lat = args.width // 16
    z_dim = DEFAULT_DIT_CONFIG["in_dim"]
    latents = torch.randn((1, z_dim, T_lat, H_lat, W_lat), dtype=torch.float32, generator=g)
    # Mirror what `Wan22Core.infer` does: I2V-style first-frame anchor — leave it as random
    # noise for this structural diff (the DiT shouldn't care about content, only token shape).
    latents = latents.to(device=device, dtype=torch_dtype)

    # Dummy context: random [1, 512, 4096]. Both models should pad/truncate it identically.
    context_len = 512
    text_dim = DEFAULT_DIT_CONFIG["text_dim"]
    context = torch.randn((1, context_len, text_dim), generator=g, dtype=torch.float32).to(device=device, dtype=torch_dtype)
    context_mask = torch.ones((1, context_len), dtype=torch.bool, device=device)
    # Official wants variable-length context list — use the full 512 tokens (no truncation).
    of_context_list = [context[0].clone()]

    timestep = torch.tensor([args.timestep], dtype=torch_dtype, device=device)

    # Per-token timestep for the official, built like wan/textimage2video.py:573-578:
    # first-frame tokens = 0, rest = timestep.
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    seq_len = T_lat * tokens_per_frame
    per_token_t = timestep[0].expand(seq_len).clone()
    per_token_t[:tokens_per_frame] = 0
    per_token_t = per_token_t.unsqueeze(0)

    # ---- 3. Hook installation ----------------------------------------------------------
    fw_traces = []  # list of (name, x_in, x_out)
    of_traces = []

    def _mk_hook(traces, name):
        def _hook(module, inp, out):
            x_in = inp[0].detach().to(torch.float32).cpu()
            x_out = out.detach().to(torch.float32).cpu() if isinstance(out, torch.Tensor) else None
            traces.append((name, x_in, x_out))
        return _hook

    fw_handles = []
    for i, block in enumerate(fw_model.blocks):
        fw_handles.append(block.register_forward_hook(_mk_hook(fw_traces, f"block_{i:02d}")))
    of_handles = []
    for i, block in enumerate(of_model.blocks):
        of_handles.append(block.register_forward_hook(_mk_hook(of_traces, f"block_{i:02d}")))

    # ---- 4. Run both forward passes ----------------------------------------------------
    print("\nRunning FastWAM forward...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
        fw_out = fw_model(
            x=latents, timestep=timestep,
            context=context, context_mask=context_mask,
            action=None, fuse_vae_embedding_in_latents=True,
        )
    print("Running official forward...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
        of_out_list = of_model([latents[0]], t=per_token_t, context=of_context_list, seq_len=seq_len)
    of_out = of_out_list[0].unsqueeze(0)

    for h in fw_handles + of_handles:
        h.remove()

    # ---- 5. Compare --------------------------------------------------------------------
    print("\n=== Final output diff ===")
    fw_out_cpu = fw_out.detach().to(torch.float32).cpu()
    of_out_cpu = of_out.detach().to(torch.float32).cpu()
    if fw_out_cpu.shape != of_out_cpu.shape:
        print(f"  SHAPE MISMATCH: FastWAM {tuple(fw_out_cpu.shape)} vs official {tuple(of_out_cpu.shape)}")
    else:
        diff = (fw_out_cpu - of_out_cpu).abs()
        print(f"  shape={tuple(fw_out_cpu.shape)}  max|Δ|={diff.max().item():.4e}  mean|Δ|={diff.mean().item():.4e}")

    print("\n=== Per-block diff (input → output) ===")
    print(f"{'block':<10s}  {'shape_in':<25s}  {'in max|Δ|':<14s}  {'shape_out':<25s}  {'out max|Δ|':<14s}")
    first_diverge: Optional[int] = None
    for i, ((fw_name, fw_in, fw_out_blk), (of_name, of_in, of_out_blk)) in enumerate(zip(fw_traces, of_traces)):
        assert fw_name == of_name, f"hook ordering mismatch: {fw_name} vs {of_name}"
        in_diff = (fw_in - of_in).abs().max().item() if fw_in.shape == of_in.shape else float("nan")
        out_diff = (fw_out_blk - of_out_blk).abs().max().item() if fw_out_blk.shape == of_out_blk.shape else float("nan")
        if args.show_each_block or i < 3 or i == len(fw_traces) - 1:
            print(f"  {fw_name:<10s}  {str(tuple(fw_in.shape)):<25s}  {in_diff:<14.4e}  "
                  f"{str(tuple(fw_out_blk.shape)):<25s}  {out_diff:<14.4e}")
        if first_diverge is None and out_diff > args.threshold:
            first_diverge = i
            if not args.show_each_block:
                print(f"  {fw_name:<10s}  {str(tuple(fw_in.shape)):<25s}  {in_diff:<14.4e}  "
                      f"{str(tuple(fw_out_blk.shape)):<25s}  {out_diff:<14.4e}  <-- FIRST DIVERGENCE")

    print("\n=== Verdict ===")
    if first_diverge is None:
        print(f"No block exceeds the divergence threshold ({args.threshold}). "
              f"Either both forward passes agree, or the threshold is too high. "
              f"Re-run with --show-each-block to inspect every layer.")
    else:
        print(f"First diverged block: blocks.{first_diverge:02d}")
        print(f"Its INPUT diff: {(fw_traces[first_diverge][1] - of_traces[first_diverge][1]).abs().max().item():.4e}")
        print(f"Its OUTPUT diff: {(fw_traces[first_diverge][2] - of_traces[first_diverge][2]).abs().max().item():.4e}")
        if first_diverge == 0:
            input_diff = (fw_traces[0][1] - of_traces[0][1]).abs().max().item()
            if input_diff < 1e-3:
                print("\nINTERPRETATION: block_00's INPUT matches between FastWAM and official, but "
                      "its OUTPUT diverges. The bug is in the DiTBlock forward itself "
                      "(attention, modulation, gate, ffn, or norm).")
            else:
                print("\nINTERPRETATION: even block_00's INPUT diverges. The bug is in `pre_dit` "
                      "(patch_embedding, time_embedding, text_embedding) or the freqs construction.")
        else:
            print(f"\nINTERPRETATION: blocks 0..{first_diverge - 1} are in agreement; the "
                  f"divergence starts at block {first_diverge}. Could be either:")
            print(f"  - error accumulation in earlier blocks finally crosses the threshold "
                  f"(check the trend in earlier blocks via --show-each-block),")
            print(f"  - or a structural difference that only matters at deeper blocks.")

    return 0 if first_diverge is None else 1


if __name__ == "__main__":
    sys.exit(main())
