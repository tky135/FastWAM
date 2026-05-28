"""Hook the entire post-Head pipeline with explicit dtype reporting.

After verify_unpatchify.py confirms the two unpatchify implementations are bit-exact
AND diff_head.py shows head/head.norm/head.head all match 0.0 — yet the FINAL output
still diverges by 7.76e-3 — there must be an implicit dtype cast happening somewhere
we haven't pinpointed. This script:
  1. Hooks the whole WanVideoDiT/WanModel (captures model.forward output).
  2. Hooks Head (input/output).
  3. Prints dtype + a sample value at every hook.
  4. Compares the final outputs and shows their dtypes too.
"""

import sys
from pathlib import Path
import argparse
import glob


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

    validated = _validate_dit_config(DEFAULT_DIT_CONFIG)
    fw_model = WanVideoDiT(**validated)
    shards = sorted(glob.glob(str(Path(ckpt_dir) / "diffusion_pytorch_model*.safetensors")))
    fw_state = load_state_dict(shards if len(shards) > 1 else shards[0],
                               torch_dtype=torch_dtype, device="cpu")
    fw_state = wan_video_dit_state_dict_converter(fw_state)
    fw_model.load_state_dict(fw_state, strict=False)
    fw_model = fw_model.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)
    of_model = WanModel.from_pretrained(ckpt_dir).to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

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

    # Captures dict: name -> dtype + sample diff vs paired tensor.
    fw_caps = {}  # name -> (dtype, value-clone-fp32-cpu)
    of_caps = {}

    def _mk_hook(caps, name):
        def _hook(module, inp, out):
            # Capture output (handle list returns from WanModel.forward).
            if isinstance(out, list):
                if out and hasattr(out[0], "detach"):
                    caps[name] = (out[0].dtype, out[0].detach().to(torch.float32).cpu())
                else:
                    caps[name] = (None, None)
            elif hasattr(out, "detach"):
                caps[name] = (out.dtype, out.detach().to(torch.float32).cpu())
            else:
                caps[name] = (None, None)
        return _hook

    # Hook ENTIRE model + head sub-modules. The empty-name hook on WanVideoDiT/WanModel
    # captures the *actual* model.forward return value.
    fw_handles = [
        fw_model.register_forward_hook(_mk_hook(fw_caps, "<model>")),
        fw_model.head.register_forward_hook(_mk_hook(fw_caps, "head")),
        fw_model.head.head.register_forward_hook(_mk_hook(fw_caps, "head.head")),
    ]
    of_handles = [
        of_model.register_forward_hook(_mk_hook(of_caps, "<model>")),
        of_model.head.register_forward_hook(_mk_hook(of_caps, "head")),
        of_model.head.head.register_forward_hook(_mk_hook(of_caps, "head.head")),
    ]

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

    print("\n=== Hook captures (with dtypes) ===")
    print(f"{'name':<14s}  {'fw dtype':<14s}  {'of dtype':<14s}  {'shape':<28s}  {'max|Δ|':<14s}  {'mean|Δ|':<14s}")
    for name in ["head.head", "head", "<model>"]:
        fw_e = fw_caps.get(name, (None, None))
        of_e = of_caps.get(name, (None, None))
        fw_dtype, fw_v = fw_e
        of_dtype, of_v = of_e
        if fw_v is None or of_v is None:
            print(f"  {name:<12s}  {'-':<14s}  {'-':<14s}  missing")
            continue
        if fw_v.shape != of_v.shape:
            print(f"  {name:<12s}  {str(fw_dtype):<14s}  {str(of_dtype):<14s}  "
                  f"SHAPE-MISMATCH fw={tuple(fw_v.shape)} of={tuple(of_v.shape)}")
            continue
        d = (fw_v - of_v).abs()
        print(f"  {name:<12s}  {str(fw_dtype):<14s}  {str(of_dtype):<14s}  "
              f"{str(tuple(fw_v.shape)):<28s}  {d.max().item():<14.4e}  {d.mean().item():<14.4e}")

    # Also report the actual returned tensors from outside the model.
    print("\n=== Returned tensors (from outside the model) ===")
    print(f"  fw_out: dtype={fw_out.dtype}, shape={tuple(fw_out.shape)}")
    print(f"  of_out: dtype={of_out.dtype}, shape={tuple(of_out.shape)}")
    fw_cpu = fw_out.detach().to(torch.float32).cpu()
    of_cpu = of_out.detach().to(torch.float32).cpu()
    d = (fw_cpu - of_cpu).abs()
    print(f"  diff: max|Δ|={d.max().item():.4e}, mean|Δ|={d.mean().item():.4e}")

    # Check: does FastWAM's <model> hook capture match the returned fw_out?
    print("\n=== Consistency check: hook capture of <model> vs returned tensor ===")
    fw_model_hook_dtype, fw_model_hook_val = fw_caps.get("<model>", (None, None))
    if fw_model_hook_val is not None:
        eq_fw = torch.equal(fw_model_hook_val, fw_cpu)
        print(f"  FastWAM: hook_capture <model> == fw_out: {eq_fw}  "
              f"(hook dtype: {fw_model_hook_dtype}, fw_out dtype: {fw_out.dtype})")
    of_model_hook_dtype, of_model_hook_val = of_caps.get("<model>", (None, None))
    if of_model_hook_val is not None:
        eq_of = torch.equal(of_model_hook_val, of_cpu)
        print(f"  Official: hook_capture <model>[0] == of_out[0]: {eq_of}  "
              f"(hook dtype: {of_model_hook_dtype}, of_out dtype: {of_out.dtype})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
