"""Verify FastWAM's `unpatchify` produces the same output as the official `WanModel.unpatchify`.

Both should map a [B, T_lat*(H_lat/2)*(W_lat/2), out_dim*prod(patch_size)] tensor to
[B, out_dim, T_lat*patch_t, H_lat*patch_h/?, W_lat*patch_w/?]. Even though my hand-derivation
shows the two implementations are mathematically equivalent, the empirical diff in
diff_head.py is 7.76e-3 at the final output despite head.head output matching 0.0 —
something is happening between them.

This script:
  1. Builds a deterministic fp32 input tensor [1, 11440, 192].
  2. Runs both unpatchify implementations.
  3. Compares element-by-element.

If they produce identical output, the bug isn't in unpatchify — it's in dtype handling
either before or after. If they differ, the bug IS in unpatchify (likely the einops
rearrange pattern or the einsum permutation).
"""

import sys
from pathlib import Path


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


def main():
    import torch
    import math

    # Configuration matching diff_head.py / play_wan.py defaults.
    T_lat = 13  # (num_frames=49 - 1) // 4 + 1
    H_lat_patched = 22  # H_lat // patch_h = (704 // 16) // 2
    W_lat_patched = 40  # W_lat // patch_w = (1280 // 16) // 2
    seq_len = T_lat * H_lat_patched * W_lat_patched  # 11440
    out_dim = 48
    patch_size = (1, 2, 2)
    inner_dim = out_dim * math.prod(patch_size)  # 192

    print(f"Test config: T_lat={T_lat}, H_patched={H_lat_patched}, W_patched={W_lat_patched}, "
          f"seq_len={seq_len}, out_dim={out_dim}, patch_size={patch_size}, inner_dim={inner_dim}\n")

    # Deterministic input.
    torch.manual_seed(42)
    x = torch.randn((1, seq_len, inner_dim), dtype=torch.float32, device="cuda")
    print(f"Input: shape={tuple(x.shape)}, dtype={x.dtype}")

    # FastWAM unpatchify.
    from einops import rearrange
    fw_out = rearrange(
        x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
        f=T_lat, h=H_lat_patched, w=W_lat_patched,
        x=patch_size[0], y=patch_size[1], z=patch_size[2],
    )
    print(f"\nFastWAM unpatchify output: shape={tuple(fw_out.shape)}, dtype={fw_out.dtype}")

    # Official unpatchify — replicated verbatim from Wan2.2/wan/modules/model.py:499-522.
    grid_sizes = torch.tensor([[T_lat, H_lat_patched, W_lat_patched]], dtype=torch.long)

    def official_unpatchify(x_t, grid_sizes_t, out_dim_v, patch_size_v):
        c = out_dim_v
        out = []
        for u, v in zip(x_t, grid_sizes_t.tolist()):
            u = u[:math.prod(v)].view(*v, *patch_size_v, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, patch_size_v)])
            out.append(u)
        return out

    of_out_list = official_unpatchify(x, grid_sizes, out_dim, patch_size)
    of_out = of_out_list[0].unsqueeze(0)
    print(f"Official unpatchify output: shape={tuple(of_out.shape)}, dtype={of_out.dtype}")

    # Compare.
    print("\n=== Comparison ===")
    if fw_out.shape != of_out.shape:
        print(f"  SHAPE MISMATCH: FastWAM {tuple(fw_out.shape)}  vs  Official {tuple(of_out.shape)}")
        return 1

    diff = (fw_out - of_out).abs()
    print(f"  max|Δ|  = {diff.max().item():.4e}")
    print(f"  mean|Δ| = {diff.mean().item():.4e}")
    print(f"  bit-exact equal? {torch.equal(fw_out, of_out)}")

    # If they differ, sample a few positions to localize.
    if not torch.equal(fw_out, of_out):
        # Find positions with max abs diff.
        print("\n=== Positions with largest diff (top 5) ===")
        flat_diff = diff.flatten()
        top_vals, top_idx = flat_diff.topk(5)
        for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
            # Convert flat index to multi-dim.
            multi = []
            remaining = idx
            for s in reversed(fw_out.shape):
                multi.append(remaining % s)
                remaining //= s
            multi.reverse()
            pos = tuple(multi)
            fw_val = fw_out[pos].item()
            of_val = of_out[pos].item()
            print(f"  pos={pos}  fw={fw_val:+.6e}  of={of_val:+.6e}  Δ={v:.4e}")

        # Try also the alternative rearrange interpretation in case einops is parsing
        # the pattern groups differently than I think.
        print("\n=== Alternative rearrange interpretations to test ===")
        # Maybe the (x y z c) order is wrong. Try (c x y z), (x y z c), etc.
        for pattern_dim2 in ["(x y z c)", "(c x y z)", "(x y z c)", "(p q r c)"]:
            try:
                pat = f'b (f h w) {pattern_dim2} -> b c (f x) (h y) (w z)'
                alt = rearrange(
                    x, pat,
                    f=T_lat, h=H_lat_patched, w=W_lat_patched,
                    x=patch_size[0], y=patch_size[1], z=patch_size[2],
                )
                if alt.shape == of_out.shape:
                    d = (alt - of_out).abs()
                    eq = torch.equal(alt, of_out)
                    print(f"  pattern '{pat}': bit-exact={eq}, max|Δ|={d.max().item():.4e}")
            except Exception as e:
                print(f"  pattern '{pat}': errored -> {e}")

    return 0 if torch.equal(fw_out, of_out) else 1


if __name__ == "__main__":
    sys.exit(main())
