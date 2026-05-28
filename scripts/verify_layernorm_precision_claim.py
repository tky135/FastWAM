"""Verify the specific claim about why `norm1` diverged in diff_block_0.py:

  Claim: at block_0, where x enters as bf16 from pre_dit's patch_embedding,
    - Official `WanLayerNorm.forward(bf16_x)` = `nn.LayerNorm(x.float()).type_as(x)` returns BF16.
    - FastWAM `nn.LayerNorm(bf16_x)` under autocast bf16 returns FP32 (per PyTorch's
      autocast policy that puts LayerNorm in the fp32-promotion list).
    - When both are cast to fp32 for comparison (as `_to_diffable` does in the diff script),
      the official side has lost ~7 bits of mantissa, FastWAM hasn't.
    - The observed `max|Δ| ≈ 3.12e-2` is the bf16 quantization noise of LayerNorm's fp32 output.

This script verifies all three points using a synthetic bf16 input. No model load required.
"""

import torch
import torch.nn as nn


def main():
    if not torch.cuda.is_available():
        print("This script requires CUDA — autocast policy is GPU-specific.")
        return 1
    device = torch.device("cuda")

    # ---------------------------------------------------------------------------------
    # Setup: same hidden_dim and seq_len as the real block_0 input. Tokens are bf16
    # because that's what pre_dit produces (Conv3d patch_embedding under autocast bf16).
    # ---------------------------------------------------------------------------------
    torch.manual_seed(42)
    hidden_dim = 3072
    seq_len = 11440  # matches the 49-frame / 1280×704 setup from diff_dit_forward.py
    eps = 1e-6

    x_bf16 = torch.randn((1, seq_len, hidden_dim), device=device, dtype=torch.bfloat16)
    print(f"Input x: shape={tuple(x_bf16.shape)}, dtype={x_bf16.dtype}")

    # ---------------------------------------------------------------------------------
    # Two LayerNorm implementations, identical math, identical params (none, since
    # elementwise_affine=False). Only the precision-handling differs.
    # ---------------------------------------------------------------------------------
    fastwam_ln = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False).to(device)

    class WanLayerNormOfficial(nn.LayerNorm):
        """Verbatim copy of `Wan2.2/wan/modules/model.py:88-98`."""
        def __init__(self, dim, eps=1e-6, elementwise_affine=False):
            super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
        def forward(self, x):
            return super().forward(x.float()).type_as(x)

    official_ln = WanLayerNormOfficial(hidden_dim, eps=eps, elementwise_affine=False).to(device)

    # ---------------------------------------------------------------------------------
    # Run both under the same autocast context as the diff script (bf16).
    # ---------------------------------------------------------------------------------
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        fw_out = fastwam_ln(x_bf16)
        of_out = official_ln(x_bf16)

    print("\n=== Output dtype check ===")
    print(f"  FastWAM nn.LayerNorm under autocast bf16  → output dtype = {fw_out.dtype}")
    print(f"  Official WanLayerNorm under autocast bf16 → output dtype = {of_out.dtype}")

    claim_fastwam_fp32 = (fw_out.dtype == torch.float32)
    claim_official_bf16 = (of_out.dtype == torch.bfloat16)
    print(f"\n  CLAIM A: FastWAM returns fp32           → {'CONFIRMED' if claim_fastwam_fp32 else 'REFUTED'}")
    print(f"  CLAIM B: Official returns bf16          → {'CONFIRMED' if claim_official_bf16 else 'REFUTED'}")

    # ---------------------------------------------------------------------------------
    # Cast both to fp32 (same as `_to_diffable` in the diff script) and compute the diff.
    # ---------------------------------------------------------------------------------
    fw_fp32 = fw_out.detach().to(torch.float32).cpu()
    of_fp32 = of_out.detach().to(torch.float32).cpu()
    observed_diff = (fw_fp32 - of_fp32).abs()
    print("\n=== Observed diff between the two LN outputs (both cast to fp32) ===")
    print(f"  max|Δ|  = {observed_diff.max().item():.4e}")
    print(f"  mean|Δ| = {observed_diff.mean().item():.4e}")

    # ---------------------------------------------------------------------------------
    # Predict the bf16 quantization noise: take FastWAM's fp32 output, manually round-trip
    # through bf16, and measure the noise this introduces. If the claim is right, this
    # noise should equal the observed diff above.
    # ---------------------------------------------------------------------------------
    fw_fp32_quantized_to_bf16 = fw_fp32.to(torch.bfloat16).to(torch.float32)
    predicted_diff = (fw_fp32 - fw_fp32_quantized_to_bf16).abs()
    print("\n=== Predicted noise: FastWAM's fp32 LN output round-tripped through bf16 ===")
    print(f"  max|Δ|  = {predicted_diff.max().item():.4e}")
    print(f"  mean|Δ| = {predicted_diff.mean().item():.4e}")

    # ---------------------------------------------------------------------------------
    # Verdict: do observed and predicted match? If yes, the claim is verified.
    # ---------------------------------------------------------------------------------
    print("\n=== Verdict ===")
    rel_diff_max = abs(observed_diff.max().item() - predicted_diff.max().item()) / max(predicted_diff.max().item(), 1e-12)
    rel_diff_mean = abs(observed_diff.mean().item() - predicted_diff.mean().item()) / max(predicted_diff.mean().item(), 1e-12)
    print(f"  Relative difference between observed and predicted max:  {rel_diff_max:.2%}")
    print(f"  Relative difference between observed and predicted mean: {rel_diff_mean:.2%}")

    # Also check: is the observed diff exactly zero modulo bf16 quantization? I.e. is
    # `fw_out.to(bf16)` bit-identical to `of_out`?
    fw_truncated_to_bf16 = fw_out.to(torch.bfloat16)
    bit_exact = torch.equal(fw_truncated_to_bf16, of_out)
    print(f"\n  FastWAM_fp32_output.to(bf16) ==bit-exact== Official_bf16_output ?  {bit_exact}")

    if (claim_fastwam_fp32 and claim_official_bf16 and bit_exact
            and rel_diff_max < 0.10 and rel_diff_mean < 0.10):
        print("\n  RESULT: CLAIM FULLY VERIFIED.")
        print("    - FastWAM returns fp32; Official returns bf16 (matches claim).")
        print("    - The observed 'divergence' between them is purely the bf16 quantization "
              "applied to FastWAM's full-precision output.")
        print("    - Mathematically, both LayerNorms compute the IDENTICAL value in fp32; only "
              "the output dtype (and hence the precision retained) differs.")
        print("    - Fix direction (replace FastWAM's nn.LayerNorm with a Wan22LayerNorm that "
              "casts back to input dtype) is correct.")
        return 0
    else:
        print("\n  RESULT: CLAIM PARTIALLY OR FULLY REFUTED.")
        print("    See per-line confirmations above. The original divergence in diff_block_0.py "
              "is NOT purely a dtype-truncation artifact — there's an additional source of "
              "divergence that this isolated test doesn't capture.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
