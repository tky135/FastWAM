from dataclasses import dataclass
import glob
import inspect
import os
from typing import Any

import torch
import time

from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_dit_state_dict_converter,
    wan_video_vae_state_dict_converter,
)
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE38
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[vae_config.origin_file_pattern]
    return dit_config, text_config, vae_config, tokenizer_config


def _load_model_from_path(
    path,
    model_class,
    torch_dtype: torch.dtype,
    device: str,
    state_dict_converter=None,
    model_kwargs: dict[str, Any] | None = None,
    label: str = "model",
):
    """Load a registered model class from an explicit path, bypassing the hash registry.

    Mirrors `_load_registered_model` minus the hash detection — used when loading from a
    known local checkpoint directory (e.g. the official Wan2.2 repo) where the file→class
    mapping is already known. Unlike `_load_registered_model`, this *logs* any
    missing/unexpected keys rather than silently swallowing them: `strict=False` is needed
    because some Wan variants ship extra tensors, but a non-empty *missing* list means the
    model is partly random-initialized — exactly the failure mode that is otherwise
    invisible and produces subtly-wrong output.
    """
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        logger.warning(
            "%s: %d missing key(s) after load from %s (these stay at RANDOM INIT). First few: %s",
            label, len(missing), path, missing[:5],
        )
    if unexpected:
        logger.info(
            "%s: %d unexpected key(s) in %s ignored (likely sibling-variant tensors). First few: %s",
            label, len(unexpected), path, unexpected[:5],
        )
    return model.to(device=device, dtype=torch_dtype)


def _load_vae_from_path(path: str, torch_dtype: torch.dtype, device: str):
    """Load the VAE from an arbitrary file path, bypassing the hash registry.

    Used when the caller wants to override the default DiffSynth-Studio bf16 mirror
    with the official fp32 `Wan2.2_VAE.pth`. Same converter as the registry path
    (prefixes every key with `model.`), no hash validation.
    """
    return _load_model_from_path(
        path, WanVideoVAE38, torch_dtype=torch_dtype, device=device,
        state_dict_converter=wan_video_vae_state_dict_converter, label="VAE",
    )


def _resolve_local_tokenizer_dir(local_dir: str) -> str:
    """Find the bundled UMT5-XXL tokenizer directory inside an official Wan2.2 repo.

    The official `Wan-AI/Wan2.2-TI2V-5B` repo ships the tokenizer under `google/umt5-xxl/`
    (sometimes the files sit directly under `google/`). Probe both.
    """
    candidates = [
        os.path.join(local_dir, "google", "umt5-xxl"),
        os.path.join(local_dir, "google"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Tokenizer directory not found under {local_dir} (looked for {candidates}). "
        "The official Wan2.2 repo bundles it under google/umt5-xxl/."
    )


def _load_from_local_dir(
    local_dir: str,
    device: str,
    torch_dtype: torch.dtype,
    tokenizer_max_len: int,
    validated_dit_config: dict[str, Any],
    skip_dit_load_from_pretrain: bool,
    load_text_encoder: bool,
    vae_dtype: torch.dtype | None,
) -> "Wan22LoadedComponents":
    """Load every component directly from a single official Wan2.2-TI2V-5B checkpoint dir.

    No DiffSynth-Studio redirect, no hash registry, no network download. Expects the
    official repo layout:
      <local_dir>/diffusion_pytorch_model*.safetensors   (DiT, bf16)
      <local_dir>/Wan2.2_VAE.pth                          (VAE, fp32)
      <local_dir>/models_t5_umt5-xxl-enc-bf16.pth         (T5 text encoder, bf16)
      <local_dir>/google/umt5-xxl/                        (tokenizer)

    The VAE defaults to fp32 (the native precision of the official `.pth`, and what the
    official pipeline uses). Pass `vae_dtype` to override.
    """
    local_dir = str(local_dir)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"`local_dir` is not a directory: {local_dir}")

    # --- DiT ---
    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_shards = sorted(glob.glob(os.path.join(local_dir, "diffusion_pytorch_model*.safetensors")))
        if not dit_shards:
            raise FileNotFoundError(
                f"No DiT shards (diffusion_pytorch_model*.safetensors) found in {local_dir}."
            )
        dit = _load_model_from_path(
            dit_shards if len(dit_shards) > 1 else dit_shards[0],
            WanVideoDiT,
            torch_dtype=torch_dtype, device=device,
            state_dict_converter=wan_video_dit_state_dict_converter,
            model_kwargs=validated_dit_config, label="DiT",
        )
        dit_path = ",".join(dit_shards)

    # --- VAE (fp32 by default — the official .pth is fp32; this is the quality win) ---
    effective_vae_dtype = vae_dtype if vae_dtype is not None else torch.float32
    vae_file = os.path.join(local_dir, "Wan2.2_VAE.pth")
    if not os.path.isfile(vae_file):
        raise FileNotFoundError(f"VAE weights not found: {vae_file}")
    logger.info("Loading VAE from official %s at dtype=%s.", vae_file, effective_vae_dtype)
    vae = _load_model_from_path(
        vae_file, WanVideoVAE38, torch_dtype=effective_vae_dtype, device=device,
        state_dict_converter=wan_video_vae_state_dict_converter, label="VAE",
    )

    # --- T5 text encoder + tokenizer ---
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        t5_file = os.path.join(local_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.isfile(t5_file):
            raise FileNotFoundError(f"T5 text-encoder weights not found: {t5_file}")
        text_encoder = _load_model_from_path(
            t5_file, WanTextEncoder, torch_dtype=torch_dtype, device=device,
            state_dict_converter=None, label="T5",
        )
        tok_dir = _resolve_local_tokenizer_dir(local_dir)
        tokenizer = HuggingfaceTokenizer(name=tok_dir, seq_len=int(tokenizer_max_len), clean="whitespace")
        text_encoder_path = t5_file
        tokenizer_path = tok_dir
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )

    return Wan22LoadedComponents(
        dit=dit, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        dit_path=dit_path, vae_path=vae_file,
        text_encoder_path=text_encoder_path, tokenizer_path=tokenizer_path,
    )


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    vae_path: str | None = None,
    vae_dtype: torch.dtype | None = None,
    local_dir: str | None = None,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    # Fast path: everything from one official Wan2.2-TI2V-5B checkpoint directory.
    # No DiffSynth-Studio redirect, no hash registry, no network. VAE defaults to fp32.
    if local_dir is not None:
        logger.info("Loading all components directly from local_dir=%s (no DiffSynth redirect).", local_dir)
        components = _load_from_local_dir(
            local_dir=local_dir,
            device=device,
            torch_dtype=torch_dtype,
            tokenizer_max_len=tokenizer_max_len,
            validated_dit_config=validated_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            vae_dtype=vae_dtype,
        )
        logger.info("Finished loading Wan2.2-TI2V-5B components from local_dir in %.2f seconds.",
                    time.time() - start)
        return components

    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    effective_vae_dtype = vae_dtype if vae_dtype is not None else torch_dtype
    if vae_path is not None:
        logger.info(
            "Loading VAE from override path (%s) at dtype=%s, bypassing the DiffSynth redirect + hash check.",
            vae_path, effective_vae_dtype,
        )
        vae = _load_vae_from_path(vae_path, torch_dtype=effective_vae_dtype, device=device)
        resolved_vae_path = vae_path
    else:
        vae: WanVideoVAE38 = _load_registered_model(
            vae_config.path, "wan_video_vae", torch_dtype=effective_vae_dtype, device=device
        )
        resolved_vae_path = str(vae_config.path)
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=resolved_vae_path,
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
