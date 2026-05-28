"""Hook every named sub-module of `blocks.0` in both DiTs and find the first one
that diverges.

Use this AFTER `diff_dit_forward.py` localizes the bug to a specific block. This script
drills into the block's internals: norm1, self_attn (and its sub-projections), gate,
norm3 + cross_attn, norm2 + ffn — and reports the first sub-module whose output diverges.

It also captures the FULL block input tuple (not just `x`) so it can compare the
modulation tensor (`t_mod` in FastWAM, `e` in official) which is passed as an extra
positional argument to the block.
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


def _to_diffable(x):
    """Make any tensor (or first element of a tuple) into a CPU fp32 tensor for diffing."""
    if isinstance(x, (tuple, list)):
        x = x[0] if x else None
    if x is None:
        return None
    if not hasattr(x, "detach"):
        return None
    import torch
    return x.detach().to(torch.float32).cpu()


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

    # Build both models.
    validated = _validate_dit_config(DEFAULT_DIT_CONFIG)
    fw_model = WanVideoDiT(**validated)
    shards = sorted(glob.glob(str(Path(ckpt_dir) / "diffusion_pytorch_model*.safetensors")))
    fw_state = load_state_dict(shards if len(shards) > 1 else shards[0],
                               torch_dtype=torch_dtype, device="cpu")
    fw_state = wan_video_dit_state_dict_converter(fw_state)
    fw_model.load_state_dict(fw_state, strict=False)
    fw_model = fw_model.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

    of_model = WanModel.from_pretrained(ckpt_dir).to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

    # Build a deterministic input.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    T_lat = (args.num_frames - 1) // 4 + 1
    H_lat = args.height // 16
    W_lat = args.width // 16
    latents = torch.randn((1, DEFAULT_DIT_CONFIG["in_dim"], T_lat, H_lat, W_lat),
                          dtype=torch.float32, generator=g).to(device=device, dtype=torch_dtype)
    text_dim = DEFAULT_DIT_CONFIG["text_dim"]
    context_len = 512
    context = torch.randn((1, context_len, text_dim), generator=g, dtype=torch.float32).to(device=device, dtype=torch_dtype)
    context_mask = torch.ones((1, context_len), dtype=torch.bool, device=device)
    timestep = torch.tensor([args.timestep], dtype=torch_dtype, device=device)
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    seq_len = T_lat * tokens_per_frame
    per_token_t = timestep[0].expand(seq_len).clone()
    per_token_t[:tokens_per_frame] = 0
    per_token_t = per_token_t.unsqueeze(0)

    # ---- Hook every sub-module of blocks.0 ---------------------------------------------
    fw_block = fw_model.blocks[0]
    of_block = of_model.blocks[0]

    fw_traces = {}  # name -> (input_first, output)
    of_traces = {}

    def _mk_hook(traces_dict, name):
        def _hook(module, inp, out):
            traces_dict[name] = (_to_diffable(inp), _to_diffable(out))
        return _hook

    fw_handles = []
    of_handles = []

    # Hook block itself (captures full input tuple at position 0 = x; we'll separately
    # capture all positional args via a wrapper below to compare t_mod/e).
    fw_handles.append(fw_block.register_forward_hook(_mk_hook(fw_traces, "<block>")))
    of_handles.append(of_block.register_forward_hook(_mk_hook(of_traces, "<block>")))

    # Hook all named sub-modules within the block.
    fw_subs = dict(fw_block.named_modules())
    of_subs = dict(of_block.named_modules())
    common_subnames = sorted(set(fw_subs.keys()) & set(of_subs.keys()) - {""})
    print(f"Hooking {len(common_subnames)} common sub-modules of blocks.0:")
    for name in common_subnames:
        print(f"  {name}")
        fw_handles.append(fw_subs[name].register_forward_hook(_mk_hook(fw_traces, name)))
        of_handles.append(of_subs[name].register_forward_hook(_mk_hook(of_traces, name)))

    # Capture the full positional-arg tuple at the block, so we can diff t_mod vs e.
    fw_full_input = {}
    of_full_input = {}

    def _full_input_hook(target):
        def _hook(module, inp, out):
            target["inp"] = tuple(_to_diffable(t) if hasattr(t, "detach") else t for t in inp)
        return _hook

    fw_handles.append(fw_block.register_forward_hook(_full_input_hook(fw_full_input)))
    of_handles.append(of_block.register_forward_hook(_full_input_hook(of_full_input)))

    # ---- Run both forward passes -------------------------------------------------------
    print("\nRunning FastWAM forward...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
        fw_model(x=latents, timestep=timestep,
                 context=context, context_mask=context_mask,
                 action=None, fuse_vae_embedding_in_latents=True)
    print("Running official forward...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
        of_model([latents[0]], t=per_token_t, context=[context[0].clone()], seq_len=seq_len)

    for h in fw_handles + of_handles:
        h.remove()

    # ---- Compare the block's POSITIONAL arguments --------------------------------------
    # FastWAM `DiTBlock.forward(self, x, context, t_mod, freqs, context_mask, self_attn_mask)`:
    #   index 0 = x, 1 = context, 2 = t_mod, 3 = freqs, 4 = context_mask, 5 = self_attn_mask
    # Official `WanAttentionBlock.forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens)`:
    #   index 0 = x, 1 = e, 2 = seq_lens, 3 = grid_sizes, 4 = freqs, 5 = context, 6 = context_lens
    print("\n=== Block positional argument diff ===")
    fw_inp = fw_full_input.get("inp", ())
    of_inp = of_full_input.get("inp", ())

    def _diff(a, b):
        if a is None or b is None:
            return f"a={a is not None} b={b is not None} (one is None)"
        if a.shape != b.shape:
            return f"SHAPE MISMATCH: {tuple(a.shape)} vs {tuple(b.shape)}"
        d = (a - b).abs()
        return f"shape={tuple(a.shape)}  max|Δ|={d.max().item():.4e}  mean|Δ|={d.mean().item():.4e}"

    if len(fw_inp) >= 1 and len(of_inp) >= 1:
        print(f"  x       (fw[0] vs of[0]):   {_diff(fw_inp[0], of_inp[0])}")
    if len(fw_inp) >= 3 and len(of_inp) >= 2:
        print(f"  t_mod/e (fw[2] vs of[1]):   {_diff(fw_inp[2], of_inp[1])}")
    if len(fw_inp) >= 4 and len(of_inp) >= 5:
        print(f"  freqs   (fw[3] vs of[4]):   {_diff(fw_inp[3], of_inp[4])}")
    if len(fw_inp) >= 2 and len(of_inp) >= 6:
        print(f"  context (fw[1] vs of[5]):   {_diff(fw_inp[1], of_inp[5])}")

    # ---- Compare each sub-module's output ----------------------------------------------
    print("\n=== Sub-module output diffs (in forward order) ===")
    # Order the sub-modules approximately by execution order: norm1, self_attn.{norm_q,q,norm_k,k,v,o},
    # norm3, cross_attn.{q,k,v,...}, norm2, ffn.{0,2}, then attn modules themselves.
    # We'll sort by structural depth and name; rough proxy for execution order is short -> long.
    order_priority = [
        "modulation",     # the Parameter (no module, won't be hooked, just for reference)
        "norm1",
        "self_attn.norm_q", "self_attn.q", "self_attn.norm_k", "self_attn.k", "self_attn.v",
        "self_attn", "self_attn.o",
        "norm3",
        "cross_attn.norm_q", "cross_attn.q", "cross_attn.norm_k", "cross_attn.k", "cross_attn.v",
        "cross_attn", "cross_attn.o",
        "norm2",
        "ffn.0", "ffn.2", "ffn",
        "gate",   # FastWAM's no-op GateModule (no hook in official)
    ]
    seen = set()
    ordered_names = []
    for prefix in order_priority:
        for name in common_subnames:
            if name == prefix or name.startswith(prefix + "."):
                if name not in seen:
                    ordered_names.append(name)
                    seen.add(name)
    # Append any remaining common names we didn't slot in.
    for name in common_subnames:
        if name not in seen:
            ordered_names.append(name)

    print(f"{'submodule':<35s}  {'in max|Δ|':<14s}  {'out max|Δ|':<14s}  notes")
    first_diverge = None
    for name in ordered_names:
        fw = fw_traces.get(name)
        of = of_traces.get(name)
        if fw is None or of is None:
            continue
        fw_in, fw_out = fw
        of_in, of_out = of
        in_d = "n/a"
        out_d = "n/a"
        notes = ""
        if fw_in is not None and of_in is not None and fw_in.shape == of_in.shape:
            in_d = f"{(fw_in - of_in).abs().max().item():.4e}"
        elif fw_in is not None and of_in is not None:
            in_d = f"shape:{tuple(fw_in.shape)}/{tuple(of_in.shape)}"
        if fw_out is not None and of_out is not None and fw_out.shape == of_out.shape:
            d_val = (fw_out - of_out).abs().max().item()
            out_d = f"{d_val:.4e}"
            if first_diverge is None and d_val > 1e-2 and name not in ("self_attn", "cross_attn", "ffn"):
                # Skip the wrapper modules — their divergence is the cumulative effect of their children.
                first_diverge = name
                notes = "  <-- FIRST sub-module divergence"
        elif fw_out is not None and of_out is not None:
            out_d = f"shape:{tuple(fw_out.shape)}/{tuple(of_out.shape)}"
        print(f"  {name:<33s}  {in_d:<14s}  {out_d:<14s}{notes}")

    print("\n=== Verdict ===")
    if first_diverge is None:
        print("No leaf sub-module crossed the threshold. Either the divergence is in the "
              "modulation arithmetic (which has no hookable sub-module — compare t_mod/e above), "
              "or one of the wrapper modules itself does extra math outside its children.")
    else:
        print(f"First diverged sub-module: blocks.0.{first_diverge}")
        if "self_attn" in first_diverge:
            print("\nThis points at the self-attention path. Most likely culprit: the `flash_attention` "
                  "wrapper differs between FastWAM (`ctx_mask`) and official (`k_lens` + `window_size`), "
                  "OR the `rope_apply` precision/shape mismatch.")
        elif "cross_attn" in first_diverge:
            print("\nThis points at the cross-attention path. Most likely culprit: the `ctx_mask` "
                  "argument in FastWAM is being misinterpreted by its `flash_attention` wrapper. "
                  "Official passes `context_lens=None` (no masking); FastWAM constructs and passes "
                  "a positive mask.")
        elif first_diverge.startswith("norm"):
            print("\nThis points at the LayerNorm. Despite my fp32 patches, the norm output diverges. "
                  "Could be a different `elementwise_affine` setting, or a different eps.")
        elif "ffn" in first_diverge:
            print("\nThis points at the FFN. Should be a simple Linear-GELU-Linear; if it diverges, "
                  "the GELU `approximate='tanh'` flag is the usual suspect.")

    return 0 if first_diverge is None else 1


if __name__ == "__main__":
    sys.exit(main())
