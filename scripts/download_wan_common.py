"""Download the two shared Wan2.x safetensors files from DiffSynth-Studio.

Self-contained — uses `modelscope` / `huggingface_hub` directly, no FastWAM imports.
Mirrors the env-var conventions FastWAM's loader uses so the files land in the same
cache layout `play_wan.py` and the training loader pick up on next run.

Env vars (same names FastWAM honors in `helpers/io.py`):
  - DIFFSYNTH_DOWNLOAD_SOURCE   "modelscope" | "huggingface"   (default: modelscope)
  - DIFFSYNTH_MODEL_BASE_PATH   /path/to/cache                  (default: ./checkpoints/)
  - DIFFSYNTH_SKIP_DOWNLOAD     "true" | "false"                (default: false)

Resulting files land at:
  <base>/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
  <base>/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors

Examples:
    # Default (ModelScope, ./checkpoints/)
    python scripts/download_wan_common.py

    # HuggingFace into a custom cache directory
    DIFFSYNTH_DOWNLOAD_SOURCE=huggingface DIFFSYNTH_MODEL_BASE_PATH=/data/wan_cache \\
        python scripts/download_wan_common.py
"""

import argparse
import glob
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_wan_common")


# Mirrors the `redirect_dict` in FastWAM's `helpers/loader.py:131-137`. Kept in sync
# manually — if the canonical names ever change upstream, update here too.
WAN_SHARED_SAFETENSORS = [
    {
        "label": "VAE (Wan2.2_VAE)",
        "repo": "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        "file": "Wan2.2_VAE.safetensors",
    },
    {
        "label": "Text encoder (UMT5-XXL bf16)",
        "repo": "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        "file": "models_t5_umt5-xxl-enc-bf16.safetensors",
    },
]


def _resolve_base_path() -> str:
    return os.environ.get("DIFFSYNTH_MODEL_BASE_PATH") or "./checkpoints/"


def _resolve_source() -> str:
    return (os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE") or "modelscope").lower()


def _resolve_skip_download() -> bool:
    return (os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD") or "false").lower() == "true"


def _download(repo: str, file_pattern: str, local_root: str, source: str) -> None:
    """Pull `file_pattern` from `repo` into `local_root` using the chosen mirror."""
    os.makedirs(local_root, exist_ok=True)
    if source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(
            repo,
            local_dir=local_root,
            allow_file_pattern=file_pattern,
            local_files_only=False,
        )
    elif source == "huggingface":
        from huggingface_hub import snapshot_download as hf_snapshot_download

        hf_snapshot_download(
            repo,
            local_dir=local_root,
            allow_patterns=file_pattern,
            local_files_only=False,
        )
    else:
        raise ValueError(
            f"DIFFSYNTH_DOWNLOAD_SOURCE must be 'modelscope' or 'huggingface', got '{source}'."
        )


def _ensure(label: str, repo: str, file_pattern: str) -> str:
    """Make sure `file_pattern` exists locally under the cache; return its resolved path."""
    base = _resolve_base_path()
    source = _resolve_source()
    skip = _resolve_skip_download()
    local_root = os.path.join(base, repo)

    already_present = bool(glob.glob(os.path.join(local_root, file_pattern)))
    if already_present:
        logger.info(f"{label}: already cached at {local_root}, skipping download.")
    elif skip:
        logger.info(f"{label}: DIFFSYNTH_SKIP_DOWNLOAD=true and file is missing — not downloading.")
    else:
        logger.info(f"{label}: downloading via {source} → {repo} / {file_pattern}")
        _download(repo, file_pattern, local_root, source)

    matches = sorted(glob.glob(os.path.join(local_root, file_pattern)))
    if not matches:
        raise FileNotFoundError(
            f"{label}: expected a file matching '{os.path.join(local_root, file_pattern)}' "
            "after the download step, but found none. Check network access, credentials, "
            "or DIFFSYNTH_SKIP_DOWNLOAD."
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Download Wan2.x shared safetensors (VAE + UMT5-XXL text encoder) from DiffSynth-Studio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.parse_args()  # no flags; everything is env-var driven for parity with FastWAM's loader

    logger.info(
        f"Cache root: {_resolve_base_path()}  |  source: {_resolve_source()}  |  "
        f"skip_download: {_resolve_skip_download()}"
    )

    for entry in WAN_SHARED_SAFETENSORS:
        path = _ensure(entry["label"], entry["repo"], entry["file"])
        logger.info(f"  ready → {path}")

    logger.info("All shared Wan2.x safetensors are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
