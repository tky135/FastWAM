"""Verify that FastWAM's `WanVideoDiT` loads every weight from the official
`Wan-AI/Wan2.2-TI2V-5B` safetensors shards.

Hypothesis being tested: `helpers/loader.py:120` uses `model.load_state_dict(state_dict, strict=False)`,
which silently swallows missing/unexpected keys. If any of the 30 DiT blocks' weights
aren't being mapped (or any key is mapped to the wrong layer), the FastWAM DiT will
produce wrong outputs *no matter what we patch in the forward pass*. This script makes
the load explicit and prints both lists.

Usage (run on the machine that has the FastWAM checkpoints):

    python scripts/verify_dit_state_dict.py

Or, if your checkpoints aren't under the default FastWAM cache:

    python scripts/verify_dit_state_dict.py --model-id Wan-AI/Wan2.2-TI2V-5B
    python scripts/verify_dit_state_dict.py --dit-shards "/path/to/diffusion_pytorch_model*.safetensors"

Exit code: 0 if both lists are empty (load is complete), 1 if anything is missing or
unexpected (load is incomplete).
"""

import argparse
import glob
import sys
from typing import List


# Same DiT config play_wan.py uses for vanilla I2V inference. Keep this in lockstep with
# `DEFAULT_DIT_CONFIG` in scripts/play_wan.py so the diagnostic mirrors the actual load.
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


def _resolve_dit_shards(model_id: str, explicit_pattern: str | None) -> List[str]:
    """Find the DiT safetensors shards. Resolution order:
      1. --dit-shards if explicitly given (glob).
      2. Use the FastWAM loader's `ModelConfig.download_if_necessary` to point at the
         cached `Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors` shards.
    """
    if explicit_pattern:
        matches = sorted(glob.glob(explicit_pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched --dit-shards={explicit_pattern!r}")
        return matches

    # Reuse the loader's path resolution so the diagnostic loads from the same place as
    # `play_wan.py` (and the live training pipeline). This requires the fastwam package
    # to be installable — same dependency assumption as play_wan.py itself.
    from fastwam.models.wan22.helpers.io import ModelConfig

    cfg = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    cfg.download_if_necessary()
    shards = sorted(glob.glob(str(cfg.path)) if not isinstance(cfg.path, list) else cfg.path)
    if not shards:
        # cfg.path may already be a concrete list of shard paths.
        if isinstance(cfg.path, list):
            shards = sorted(cfg.path)
        else:
            shards = sorted(glob.glob(str(cfg.path) + "*"))
    if not shards:
        raise FileNotFoundError(
            f"Couldn't resolve DiT shards for model_id={model_id!r} via FastWAM's ModelConfig. "
            "Pass --dit-shards explicitly."
        )
    return shards


def main():
    parser = argparse.ArgumentParser(
        description="Check FastWAM's WanVideoDiT state-dict load for missing/unexpected keys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument(
        "--dit-shards", default=None,
        help="Glob pointing at the official DiT safetensors shards. If unset, resolved via "
             "FastWAM's ModelConfig (same path play_wan.py uses).",
    )
    parser.add_argument("--show", type=int, default=30,
                        help="Number of missing/unexpected keys to print per category.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Load dtype (matches play_wan.py's --dtype default).")
    args = parser.parse_args()

    import torch  # heavy import — defer
    from fastwam.models.wan22.helpers.io import load_state_dict
    from fastwam.models.wan22.helpers.loader import _validate_dit_config
    from fastwam.models.wan22.helpers.state_dict_converters import (
        wan_video_dit_state_dict_converter,
    )
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    shards = _resolve_dit_shards(args.model_id, args.dit_shards)
    print(f"Loading state dict from {len(shards)} shard(s):")
    for s in shards:
        print(f"  {s}")

    state_dict = load_state_dict(shards if len(shards) > 1 else shards[0],
                                 torch_dtype=torch_dtype, device="cpu")
    print(f"\nRaw state dict: {len(state_dict)} tensors before conversion.")

    converted = wan_video_dit_state_dict_converter(state_dict)
    print(f"After `wan_video_dit_state_dict_converter`: {len(converted)} tensors.")
    dropped = len(state_dict) - len(converted)
    if dropped > 0:
        # The converter filters out vace/pose/face/motion families. These should all be
        # zero for Wan2.2-TI2V-5B; flag any that are dropped for transparency.
        dropped_keys = sorted(set(state_dict.keys()) - set(converted.keys()))
        print(f"  Converter dropped {dropped} keys (filtered before load):")
        for k in dropped_keys[: args.show]:
            print(f"    {k}")
        if len(dropped_keys) > args.show:
            print(f"    ... +{len(dropped_keys) - args.show} more")

    # Build the model with the same config play_wan.py uses, then load.
    print("\nInstantiating WanVideoDiT(**DEFAULT_DIT_CONFIG)...")
    validated = _validate_dit_config(DEFAULT_DIT_CONFIG)
    model = WanVideoDiT(**validated)
    model_keys = set(model.state_dict().keys())
    file_keys = set(converted.keys())
    print(f"  Model has {len(model_keys)} parameter/buffer tensors.")
    print(f"  File has {len(file_keys)} tensors (after conversion).")

    # Run the actual load with strict=False to capture both lists.
    incompatible = model.load_state_dict(converted, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    print("\n=== Verdict ===")
    print(f"missing_keys:    {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")

    if missing:
        print(f"\nMISSING (model layers NOT initialized from file — these stay at random init!):")
        for k in missing[: args.show]:
            shape = tuple(model.state_dict()[k].shape)
            print(f"  {k}  shape={shape}")
        if len(missing) > args.show:
            print(f"  ... +{len(missing) - args.show} more")

    if unexpected:
        print(f"\nUNEXPECTED (file tensors NOT mapped into the model — silently dropped):")
        for k in unexpected[: args.show]:
            shape = tuple(converted[k].shape)
            print(f"  {k}  shape={shape}")
        if len(unexpected) > args.show:
            print(f"  ... +{len(unexpected) - args.show} more")

    # Also sanity-check shape matches for keys that DID match (load_state_dict only
    # reports shape mismatches via warnings if size_mismatch is checked separately, but
    # strict=False suppresses them too).
    matched = model_keys & file_keys
    shape_mismatches = []
    for k in sorted(matched):
        m_shape = tuple(model.state_dict()[k].shape)
        f_shape = tuple(converted[k].shape)
        if m_shape != f_shape:
            shape_mismatches.append((k, m_shape, f_shape))
    if shape_mismatches:
        print(f"\nSHAPE MISMATCHES on {len(shape_mismatches)} matched keys "
              f"(model expects different shape than file provides):")
        for k, m_shape, f_shape in shape_mismatches[: args.show]:
            print(f"  {k}  model={m_shape}  file={f_shape}")
        if len(shape_mismatches) > args.show:
            print(f"  ... +{len(shape_mismatches) - args.show} more")

    incomplete = bool(missing) or bool(shape_mismatches)
    print()
    if incomplete:
        print("RESULT: load is INCOMPLETE — see lists above. This explains why "
              "forward-pass patches couldn't close the gap with the official DiT.")
        return 1
    elif unexpected:
        print("RESULT: load is COMPLETE for the model (no missing keys, no shape mismatches), "
              "but the file has extra unused tensors. These are harmless — likely from sibling "
              "Wan variants. Hypothesis #1 is NOT the bug; look elsewhere.")
        return 0
    else:
        print("RESULT: load is FULLY CLEAN — no missing, no unexpected, no shape mismatches. "
              "Hypothesis #1 is NOT the bug; look elsewhere (most likely the flash_attention "
              "wrapper or rope_apply precision).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
