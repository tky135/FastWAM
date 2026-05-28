"""Drill into the post-block path: Head and unpatchify.

After diff_dit_forward.py shows blocks all match (0.0 across all 30 blocks) but the
final output still diverges, the bug has to be between block_29's output and the
final returned tensor — namely Head.forward and the unpatchify reshape.

This script:
  1. Loads both DiTs with the same weights.
  2. Builds deterministic inputs (random latent, random context, fixed timestep).
  3. Runs both forwards, hooking:
       - Head's input (= block_29 output) and output (= post-Head tensor).
       - Head's `norm` and `head` (Linear) sub-modules to localize within Head.
       - Captures the unflatten/unpatchify result to compare reshape correctness.
  4. Reports each diff.

If `Head.norm` output matches but `Head.head` (the final Linear) output diverges,
the bug is in the head Linear's precision policy. If `Head.norm` output already
diverges, the bug is in the norm or modulation arithmetic inside Head.
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import Optional


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
    candidates = []
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
    raise ImportError("Cannot find the official Wan2.2 `wan` package.")


def _resolve_ckpt_dir(explicit, model_id):
    if explicit:
        return explicit
    from fastwam.models.wan22.helpers.io import ModelConfig
    cfg = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    cfg.download_if_necessary()
    path = cfg.path
    if isinstance(path, list):
        path = path[0]
    return str(Path(path).parent)


def _to_diff(t):
    if t is None:
        return None
    if isinstance(t, (tuple, list)):
        if not t:
            return None
        t = t[0]
    if not hasattr(t, "detach"):
        return None
    import torch
    return t.detach().to(torch.float32).cpu()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--timestep", type=float, default=500.0)
    args = parser.parse_args()

    import torch
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
    print(f"Checkpoint dir: {ckpt_dir}\n")

    validated = _validate_dit_config(DEFAULT_DIT_CONFIG)
    fw_model = WanVideoDiT(**validated)
    shards = sorted(glob.glob(str(Path(ckpt_dir) / "diffusion_pytorch_model*.safetensors")))
    fw_state = load_state_dict(shards if len(shards) > 1 else shards[0],
                               torch_dtype=torch_dtype, device="cpu")
    fw_state = wan_video_dit_state_dict_converter(fw_state)
    fw_model.load_state_dict(fw_state, strict=False)
    fw_model = fw_model.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)
    of_model = WanModel.from_pretrained(ckpt_dir).to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

    # Deterministic inputs (use fp32 timestep to match production).
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    T_lat = (args.num_frames - 1) // 4 + 1
    H_lat = args.height // 16
    W_lat = args.width // 16
    latents = torch.randn((1, DEFAULT_DIT_CONFIG["in_dim"], T_lat, H_lat, W_lat),
                          dtype=torch.float32, generator=g).to(device=device, dtype=torch_dtype)
    context_len = 512
    context = torch.randn((1, context_len, DEFAULT_DIT_CONFIG["text_dim"]),
                          generator=g, dtype=torch.float32).to(device=device, dtype=torch_dtype)
    context_mask = torch.ones((1, context_len), dtype=torch.bool, device=device)
    timestep = torch.tensor([args.timestep], dtype=torch.float32, device=device)
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    seq_len = T_lat * tokens_per_frame
    per_token_t = timestep[0].expand(seq_len).clone()
    per_token_t[:tokens_per_frame] = 0
    per_token_t = per_token_t.unsqueeze(0)

    # Hook the Head and its sub-modules (.norm, .head) on both models.
    fw_traces = {}
    of_traces = {}

    def _mk_hook(traces, name):
        def _hook(module, inp, out):
            traces[name] = (_to_diff(inp), _to_diff(out))
        return _hook

    fw_handles, of_handles = [], []
    for name, m in fw_model.named_modules():
        if name in ("head", "head.norm", "head.head"):
            fw_handles.append(m.register_forward_hook(_mk_hook(fw_traces, name)))
    for name, m in of_model.named_modules():
        if name in ("head", "head.norm", "head.head"):
            of_handles.append(m.register_forward_hook(_mk_hook(of_traces, name)))

    # Forwards.
    print("Running FastWAM forward...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
        fw_out = fw_model(x=latents, timestep=timestep,
                          context=context, context_mask=context_mask,
                          action=None, fuse_vae_embedding_in_latents=True)
    print("Running official forward...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
        of_out_list = of_model([latents[0]], t=per_token_t, context=[context[0].clone()],
                               seq_len=seq_len)
    of_out = of_out_list[0].unsqueeze(0)

    for h in fw_handles + of_handles:
        h.remove()

    def _diff_pair(name):
        fw = fw_traces.get(name)
        of = of_traces.get(name)
        if fw is None or of is None:
            return f"{name}: missing in {'fw' if fw is None else 'of'}"
        fw_in, fw_out_v = fw
        of_in, of_out_v = of
        def _fmt(a, b):
            if a is None or b is None:
                return f"a={'set' if a is not None else 'None'} b={'set' if b is not None else 'None'}"
            if a.shape != b.shape:
                return f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
            d = (a - b).abs()
            return f"shape={tuple(a.shape)}  max|Δ|={d.max().item():.4e}  mean|Δ|={d.mean().item():.4e}"
        return (
            f"  {name}\n"
            f"      INPUT : {_fmt(fw_in, of_in)}\n"
            f"      OUTPUT: {_fmt(fw_out_v, of_out_v)}"
        )

    print("\n=== Head sub-module diffs ===")
    print(_diff_pair("head"))         # whole head (input = block_29 out, output = post-head tensor)
    print(_diff_pair("head.norm"))    # the WanLayerNorm
    print(_diff_pair("head.head"))    # the final Linear

    # And compare the FINAL post-unpatchify output for completeness.
    fw_out_cpu = fw_out.detach().to(torch.float32).cpu()
    of_out_cpu = of_out.detach().to(torch.float32).cpu()
    diff = (fw_out_cpu - of_out_cpu).abs()
    print("\n=== Final output (post-unpatchify) ===")
    print(f"  shape={tuple(fw_out_cpu.shape)}  max|Δ|={diff.max().item():.4e}  "
          f"mean|Δ|={diff.mean().item():.4e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
