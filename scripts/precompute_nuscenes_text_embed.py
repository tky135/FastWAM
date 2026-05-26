"""One-shot: cache the single fixed T5 embedding used by the nuScenes WAM dataset.

The nuScenes dataset config uses a single fixed `override_instruction`, so we only need
one cached T5 embedding (instead of the per-task cache that LIBERO/RoboTwin needs via
``scripts/precompute_text_embeds.py``). This script writes that one file with the same
filename convention ``RobotVideoDataset._get_cached_text_context`` expects:
``{sha256(prompt)}.t5_len{context_len}.wan22ti2v5b.pt``.

Run from the FastWAM repo root after the nuScenes data config is in place:

    python scripts/precompute_nuscenes_text_embed.py
"""

import hashlib
from pathlib import Path

import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

INSTRUCTION = "Front-view driving scene from a moving vehicle."
CACHE_DIR = Path("./data/text_embeds_cache/nuscenes")
CONTEXT_LEN = 128
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_ID = "Wan-AI/Wan2.1-T2V-1.3B"


def main() -> None:
    prompt = DEFAULT_PROMPT.format(task=INSTRUCTION)

    _, text_cfg, _, tok_cfg = _resolve_configs(
        model_id=MODEL_ID,
        tokenizer_model_id=TOKENIZER_ID,
        redirect_common_files=True,
    )
    text_cfg.download_if_necessary()
    tok_cfg.download_if_necessary()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = _load_registered_model(
        text_cfg.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tok_cfg.path, seq_len=CONTEXT_LEN, clean="whitespace")

    with torch.no_grad():
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ctx = encoder(ids.to(device), mask.to(device=device, dtype=torch.bool))

    payload = {
        "context": ctx[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "mask": mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous(),
    }
    import ipdb ; ipdb.set_trace()
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    out_path = CACHE_DIR / f"{hashed}.t5_len{CONTEXT_LEN}.wan22ti2v5b.pt"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
