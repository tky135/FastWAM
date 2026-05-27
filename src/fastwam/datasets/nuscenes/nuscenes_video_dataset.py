import hashlib
import os
import traceback
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from accelerate import PartialState

from fastwam.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from fastwam.utils import misc
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


class NuScenesVideoDataset(torch.utils.data.Dataset):
    """nuScenes WAM dataset, front-camera only, ego-trajectory waypoint actions.

    Returns the same dict contract as `RobotVideoDataset`:
        video:          [C=3, T_video, H, W], float32, range [-1, 1]
        action:         [T_action, 3], float32 (normalized by processor)
        proprio:        [T_action, 4], float32 (normalized by processor)
        prompt:         str
        context:        [context_len, D], float32 (T5 cache)
        context_mask:   [context_len], bool
        image_is_pad:   [T_video], bool
        action_is_pad:  [T_action], bool
        proprio_is_pad: [T], bool (mirrors RobotVideoDataset's length asymmetry)
    """

    def __init__(
        self,
        dataroot: str,
        shape_meta: DictConfig | dict,
        version: str = "v1.0-mini",
        split: str = "train",
        camera_key: str = "CAM_FRONT",
        num_frames: int = 9,
        action_video_freq_ratio: int = 1,
        video_size: list[int] = (224, 400),
        global_sample_stride: int = 1,
        processor: Optional[DictConfig | Any] = None,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        pretrained_norm_stats: Optional[str] = None,
        is_training_set: bool = False,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        override_instruction: str = "Front-view driving scene from a moving vehicle.",
    ):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        assert (num_frames - 1) % action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, "
            f"got {num_frames - 1} and {action_video_freq_ratio}"
        )
        assert ((num_frames - 1) // action_video_freq_ratio) % 4 == 0, (
            f"video frames must be divisible by 4 for VAE tokenization, "
            f"got {(num_frames - 1) // action_video_freq_ratio}"
        )

        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.camera_key = camera_key
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.video_size = list(video_size)
        self.global_sample_stride = max(1, int(global_sample_stride))
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.override_instruction = override_instruction
        self.is_training_set = is_training_set

        if isinstance(shape_meta, DictConfig):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.shape_meta = shape_meta
        self._action_key = shape_meta["action"][0]["key"]
        self._state_key = shape_meta["state"][0]["key"]
        self._image_key = shape_meta["images"][0]["key"]
        assert self._image_key == camera_key, (
            f"shape_meta.images[0].key ({self._image_key}) must equal camera_key ({camera_key})."
        )

        self.video_sample_indices = list(range(0, num_frames, action_video_freq_ratio))

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        scene_names = self._resolve_split_scenes(version, split, create_splits_scenes)
        self.sample_tokens = self._build_sample_index(scene_names)
        self._num_scenes_in_split = len(scene_names)
        logger.info(
            f"NuScenesVideoDataset[{split}]: version={version} "
            f"scenes={len(scene_names)} anchors={len(self.sample_tokens)} "
            f"num_frames={num_frames} T_video={len(self.video_sample_indices)}"
        )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self.processor = None
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if pretrained_norm_stats is None:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets."
                    )
                if PartialState().is_main_process:
                    logger.info("Computing nuScenes dataset stats for normalization...")
                    dataset_stats = self._compute_dataset_stats(processor)
                    work_dir = misc.get_work_dir() or "."
                    os.makedirs(work_dir, exist_ok=True)
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Loaded nuScenes dataset stats: {pretrained_norm_stats}")
                # Mirror into work_dir only if the source path is different — otherwise
                # rank 0's re-write races with rank N's read on the same file and
                # corrupts the JSON for the slow reader. See the runtime.build_datasets
                # path: pretrained_norm_stats for val is exactly work_dir/dataset_stats.json,
                # which makes the re-save pure redundancy. Skipping it eliminates the race.
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir() or "."
                    target_path = os.path.realpath(os.path.join(work_dir, "dataset_stats.json"))
                    source_path = os.path.realpath(pretrained_norm_stats)
                    if source_path != target_path:
                        os.makedirs(work_dir, exist_ok=True)
                        save_dataset_stats_to_json(dataset_stats, target_path)

            processor.set_normalizer_from_stats(dataset_stats)
            if is_training_set:
                processor.train()
            else:
                processor.eval()
            self.processor = processor

        # Shim so trainer eval path (`val_dataset.lerobot_dataset.processor`) doesn't AttributeError
        # if eval ever fires for this dataset. The driving eval is a follow-up PR.
        self.lerobot_dataset = self

    # ---- index construction ----

    def _resolve_split_scenes(self, version: str, split: str, create_splits_scenes) -> list[str]:
        all_splits = create_splits_scenes()
        is_mini_version = version.endswith("mini")
        if split == "mini":
            # Combined v1.0-mini set: mini_train + mini_val (no internal train/val split).
            wanted = set(all_splits.get("mini_train", [])) | set(all_splits.get("mini_val", []))
            resolved = "mini_train+mini_val"
        elif split in all_splits:
            # Direct nuScenes split key, e.g. "mini_train", "mini_val", "test".
            wanted = set(all_splits[split])
            resolved = split
        elif split == "train":
            resolved = "mini_train" if is_mini_version else "train"
            wanted = set(all_splits[resolved])
        elif split == "val":
            resolved = "mini_val" if is_mini_version else "val"
            wanted = set(all_splits[resolved])
        else:
            raise ValueError(
                f"Unknown split '{split}'. Known: {sorted(all_splits)} + 'mini' alias."
            )
        present = {s["name"]: s["token"] for s in self.nusc.scene}
        scene_names = sorted(wanted & set(present.keys()))
        if not scene_names:
            raise RuntimeError(
                f"No scenes from split '{resolved}' present in dataroot '{self.dataroot}'."
            )
        return scene_names

    def _build_sample_index(self, scene_names: list[str]) -> list[str]:
        # Anchors must have at least ONE prior keyframe in the same scene so that
        # `_compute_proprio` can compute the anchor's velocity by backward-differencing
        # against the previous frame (avoiding a future leak — see _compute_proprio).
        # Hence the range starts at 1, not 0. Each scene loses one candidate anchor.
        anchors: list[str] = []
        for s in self.nusc.scene:
            if s["name"] not in scene_names:
                continue
            tokens: list[str] = []
            tok = s["first_sample_token"]
            while tok:
                tokens.append(tok)
                tok = self.nusc.get("sample", tok)["next"]
            max_anchor = len(tokens) - self.num_frames + 1
            if max_anchor <= 1:
                continue
            for i in range(1, max_anchor, self.global_sample_stride):
                anchors.append(tokens[i])
        return anchors

    def __len__(self) -> int:
        return len(self.sample_tokens)

    # ---- nuScenes IO ----

    def _walk_forward_keyframes(self, anchor_sample_token: str, n: int) -> list[dict]:
        out: list[dict] = []
        tok = anchor_sample_token
        for _ in range(n):
            if not tok:
                raise RuntimeError(
                    f"Ran out of keyframes walking forward from {anchor_sample_token} after {len(out)}"
                )
            sample = self.nusc.get("sample", tok)
            out.append(sample)
            tok = sample["next"]
        return out

    def _load_front_image(self, sample: dict) -> torch.Tensor:
        sd_token = sample["data"][self.camera_key]
        sd = self.nusc.get("sample_data", sd_token)
        path = os.path.join(self.dataroot, sd["filename"])
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)  # [H, W, 3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]
        return tensor

    def _get_ego_pose(self, sample: dict) -> tuple[np.ndarray, np.ndarray]:
        sd = self.nusc.get("sample_data", sample["data"][self.camera_key])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        return (
            np.asarray(ep["translation"], dtype=np.float64),
            np.asarray(ep["rotation"], dtype=np.float64),
        )

    @staticmethod
    def _yaw_from_quat(rotations: np.ndarray) -> np.ndarray:
        from pyquaternion import Quaternion

        yaws = np.empty(rotations.shape[0], dtype=np.float64)
        for i, q in enumerate(rotations):
            yaws[i] = Quaternion(q[0], q[1], q[2], q[3]).yaw_pitch_roll[0]
        return yaws

    def _compute_waypoints(
        self, translations: np.ndarray, rotations: np.ndarray
    ) -> torch.Tensor:
        yaws = self._yaw_from_quat(rotations)
        x0, y0 = translations[0, 0], translations[0, 1]
        yaw0 = yaws[0]
        cos0, sin0 = np.cos(-yaw0), np.sin(-yaw0)
        dx_world = translations[1:, 0] - x0
        dy_world = translations[1:, 1] - y0
        dx_local = cos0 * dx_world - sin0 * dy_world
        dy_local = sin0 * dx_world + cos0 * dy_world
        dyaw = _wrap_to_pi(yaws[1:] - yaw0)
        waypoints = np.stack([dx_local, dy_local, dyaw], axis=-1).astype(np.float32)
        return torch.from_numpy(waypoints)

    def _compute_proprio(
        self,
        translations: np.ndarray,        # [T, 3], window keyframes 0..T-1
        rotations: np.ndarray,           # [T, 4], window keyframes 0..T-1
        samples: list[dict],             # length T, window keyframes 0..T-1
        prev_translation: np.ndarray,    # [3], the keyframe immediately before window[0]
        prev_rotation: np.ndarray,       # [4], same
        prev_timestamp_us: float,        # microseconds, same
    ) -> torch.Tensor:
        """Compute per-step ego dynamics using BACKWARD differencing.

        For each window step k in 0..T-1, the velocity at window-time k uses positions
        at window-times k-1 and k (where k-1=-1 means the prior keyframe). This avoids
        leaking future positions into the proprio conditioning — a model that sees
        proprio[k] cannot trivially recover the unseen position at window-time k+1.
        """
        T = translations.shape[0]
        # Extend by one step into the past so window step 0 has a valid backward neighbor.
        ext_translations = np.vstack([prev_translation[None, :], translations])  # [T+1, 3]
        ext_rotations = np.vstack([prev_rotation[None, :], rotations])           # [T+1, 4]
        ext_timestamps_us = np.concatenate(
            [[prev_timestamp_us], np.asarray([s["timestamp"] for s in samples], dtype=np.float64)]
        )                                                                        # [T+1]

        # dt[k] = window-time-k timestamp minus window-time-(k-1) timestamp, in seconds.
        dt_back = (ext_timestamps_us[1:] - ext_timestamps_us[:-1]) * 1e-6        # [T]
        dt_back = np.where(dt_back <= 1e-6, 1e-3, dt_back)  # guard against zero/dup timestamps

        # World-frame velocity at window-time k, via backward diff.
        v_world_x = (ext_translations[1:, 0] - ext_translations[:-1, 0]) / dt_back  # [T]
        v_world_y = (ext_translations[1:, 1] - ext_translations[:-1, 1]) / dt_back  # [T]

        # Rotate into the ego frame *at* window-time k (not k-1) — proprio is "your
        # current velocity expressed in your current heading".
        yaws = self._yaw_from_quat(rotations)                                       # [T]
        cos_y = np.cos(-yaws)
        sin_y = np.sin(-yaws)
        v_x = cos_y * v_world_x - sin_y * v_world_y
        v_y = sin_y * v_world_x + cos_y * v_world_y
        speed = np.sqrt(v_x * v_x + v_y * v_y)

        # Yaw-rate at window-time k, also via backward diff.
        ext_yaws = self._yaw_from_quat(ext_rotations)                               # [T+1]
        yaw_rate = _wrap_to_pi(ext_yaws[1:] - ext_yaws[:-1]) / dt_back              # [T]

        proprio = np.stack([v_x, v_y, yaw_rate, speed], axis=-1).astype(np.float32)
        return torch.from_numpy(proprio)

    def _build_raw_sample(self, idx: int) -> dict:
        anchor_token = self.sample_tokens[idx]
        samples = self._walk_forward_keyframes(anchor_token, self.num_frames)
        # Backward-diff proprio needs one keyframe before the anchor. The index builder
        # guarantees anchor index >= 1 within the scene, so anchor["prev"] is always set.
        anchor_sample = samples[0]
        prev_tok = anchor_sample["prev"]
        assert prev_tok, (
            f"anchor {anchor_token} unexpectedly has no prev keyframe; "
            "the sample index should not include scene-first keyframes."
        )
        prev_sample = self.nusc.get("sample", prev_tok)
        prev_translation, prev_rotation = self._get_ego_pose(prev_sample)
        prev_timestamp_us = float(prev_sample["timestamp"])

        imgs = torch.stack(
            [self._load_front_image(s) for s in samples], dim=0
        )  # [T, 3, H, W] uint8
        translations = np.stack([self._get_ego_pose(s)[0] for s in samples], axis=0)
        rotations = np.stack([self._get_ego_pose(s)[1] for s in samples], axis=0)
        actions = self._compute_waypoints(translations, rotations)  # [T-1, 3]
        proprio = self._compute_proprio(
            translations,
            rotations,
            samples,
            prev_translation=prev_translation,
            prev_rotation=prev_rotation,
            prev_timestamp_us=prev_timestamp_us,
        )  # [T, 4]

        T = self.num_frames
        return {
            "images": {self.camera_key: imgs},
            "action": {self._action_key: actions},
            "state": {self._state_key: proprio},
            "image_is_pad": torch.zeros(T, dtype=torch.bool),
            "action_is_pad": torch.zeros(T - 1, dtype=torch.bool),
            "state_is_pad": torch.zeros(T, dtype=torch.bool),
            "task": self.override_instruction,
            "idx": idx,
        }

    # ---- stats ----

    def _compute_dataset_stats(self, processor) -> dict:
        """One-pass scan over every anchor sample, producing the normalization stats dict
        that ``FastWAMProcessor.set_normalizer_from_stats`` consumes.

        The schema (per-key ``global_*`` and ``stepwise_*`` for min/max/mean/std/q01/q99,
        plus top-level ``num_episodes`` / ``num_transition``) mirrors
        ``BaseLerobotDataset.get_dataset_stats`` so the same JSON serializer and
        ``LinearNormalizer`` can read it. The normalizer's ``norm_default_mode`` decides
        which two of those stats are actually used at runtime (z-score → mean/std,
        min/max → global_min/max, q01/q99 → global_q01/q99).
        """
        from tqdm import tqdm

        # We collect raw action and state tensors here. We do NOT touch the normalizer
        # (it doesn't exist yet — its construction is what these stats feed into).
        action_list: list[torch.Tensor] = []
        state_list: list[torch.Tensor] = []
        for i in tqdm(range(len(self)), desc="nuScenes stats pass"):
            raw = self._build_raw_sample(i)
            # Apply only the processor's `action_state_transform` step. This validates
            # shape_meta and runs any configured action/state transforms — but does NOT
            # normalize or merge keys. We want stats over the same values the runtime
            # normalizer will see (post-transform, pre-normalize), so this stays in sync
            # if you ever add e.g. a delta-action transform.
            transformed = processor.action_state_transform(
                {
                    "action": {k: v.clone() for k, v in raw["action"].items()},
                    "state": {k: v.clone() for k, v in raw["state"].items()},
                }
            )
            action_list.append(transformed["action"][self._action_key])
            state_list.append(transformed["state"][self._state_key])

        # Stack across samples → [N, T_horizon, dim]. "stepwise" reductions keep the
        # timestep axis (T_horizon, dim); "global" reductions flatten across samples AND
        # timesteps to give just (dim,).
        action = torch.stack(action_list, dim=0)  # [N, T-1, A]
        state = torch.stack(state_list, dim=0)  # [N, T, S]

        def _per_key_stats(data: torch.Tensor) -> dict:
            # data: [N, T_horizon, dim]
            data = data.float()
            # Stepwise: reduce over the sample axis only. Shape (T_horizon, dim).
            # Used when norm mode is configured as "stepwise" — one scale/offset per timestep.
            stepwise_min = data.amin(dim=0)
            stepwise_max = data.amax(dim=0)
            stepwise_mean = data.mean(dim=0)
            stepwise_std = data.std(dim=0, unbiased=False)
            stepwise_q01 = torch.quantile(data, 0.01, dim=0)
            stepwise_q99 = torch.quantile(data, 0.99, dim=0)
            # Global: collapse samples + timesteps. Shape (dim,). The standard path
            # (z-score / min-max with `use_stepwise_action_norm: false`) uses these.
            flat = data.reshape(-1, data.shape[-1])
            global_min = flat.amin(dim=0)
            global_max = flat.amax(dim=0)
            global_mean = flat.mean(dim=0)
            global_std = flat.std(dim=0, unbiased=False)
            global_q01 = torch.quantile(flat, 0.01, dim=0)
            global_q99 = torch.quantile(flat, 0.99, dim=0)
            return {
                "stepwise_min": stepwise_min,
                "stepwise_max": stepwise_max,
                "stepwise_mean": stepwise_mean,
                "stepwise_std": stepwise_std,
                "stepwise_q01": stepwise_q01,
                "stepwise_q99": stepwise_q99,
                "global_min": global_min,
                "global_max": global_max,
                "global_mean": global_mean,
                "global_std": global_std,
                "global_q01": global_q01,
                "global_q99": global_q99,
            }

        return {
            "state": {self._state_key: _per_key_stats(state)},
            "action": {self._action_key: _per_key_stats(action)},
            "num_episodes": self._num_scenes_in_split,
            "num_transition": int(action.shape[0] * action.shape[1]),
        }

    # ---- text cache ----

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_nuscenes_text_embed.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]}"
            )
        return context, context_mask

    # ---- main __getitem__ path ----

    def _get(self, idx: int) -> dict:
        raw = self._build_raw_sample(idx)
        sample = self.processor.preprocess(raw)

        video = sample["pixel_values"]  # [num_cameras=1, T, C, H, W]
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            video = video.squeeze(0)  # [T_video, C, H, W]
        else:
            assert video.ndim == 4, (
                f"Expected pixel_values [num_cameras, T, C, H, W] or [T, C, H, W], got {video.shape}"
            )
            video = video[self.video_sample_indices, :, :, :]

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)  # [C, T_video, H, W]

        instruction = DEFAULT_PROMPT.format(task=sample["instruction"])
        context, context_mask = self._get_cached_text_context(instruction)
        context = context.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        image_is_pad = sample["image_is_pad"]
        image_is_pad = image_is_pad[self.video_sample_indices]

        return {
            # Front-camera video clip. [C=3, T_video=9, H=224, W=400], float32 in [-1, 1].
            # Frame 0 is the anchor (current observation); frames 1..8 are 0.5 s apart.
            "video": video,
            # Future ego trajectory: 8 cumulative waypoints from the anchor, in its ego
            # frame. Each row is (Δx forward, Δy left, Δyaw CCW), z-score normalized.
            # This is the model's prediction target.
            "action": sample["action"],
            # Current ego dynamics conditioning. [T_action=8, 4] = (v_x, v_y, yaw_rate,
            # speed), z-score normalized. Model only consumes row 0 (the anchor's
            # state); higher rows are present for API parity with LIBERO/RoboTwin.
            "proprio": sample["proprio"][:-1, :],
            # Final templated prompt string — the literal text that was T5-encoded into
            # `context`. Same value for every nuScenes sample (single fixed prompt).
            "prompt": instruction,
            # Pre-computed UMT5-XXL text embedding of `prompt`. [context_len=128, 4096]
            # bf16. Cross-attended by the video DiT. Constant across all samples.
            "context": context,
            # Attention mask for `context`. Always all-True after the Wan2.2 convention
            # of zeroing padded positions instead of masking them out (see `_get`).
            "context_mask": context_mask,
            # Per-video-frame padding flag. [T_video=9], bool. All-False here because
            # the index builder only keeps anchors whose 9-frame window is fully inside
            # one scene. The model uses it to skip loss on padded frames.
            "image_is_pad": image_is_pad,
            # Per-action-step padding flag. [T_action=8], bool. All-False for the same
            # reason. Masks the action regression loss at padded steps.
            "action_is_pad": sample["action_is_pad"],
            # Per-proprio-step padding flag. [T=9], bool. Length intentionally asymmetric
            # with `proprio` (which is [:-1, :]) — mirrors RobotVideoDataset's behavior.
            "proprio_is_pad": sample["proprio_is_pad"],
        }

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._get(idx)
        except Exception as exc:
            print(f"Error processing nuScenes sample idx {idx}: {exc}")
            print(traceback.format_exc())
            return self._get(int(np.random.randint(len(self))))

    # ---- eval-time visualization hook ----

    def save_eval_visualization(
        self,
        eval_dir: str,
        step_tag: str,
        pred_action_denorm,
        gt_action_denorm,
    ) -> Optional[str]:
        """Driving-specific eval visualization hook called by `Wan22Trainer.evaluate()`.

        Receives already-denormalized predicted and ground-truth ego trajectories of shape
        ``[T_action, 3]`` (rows are anchor-frame waypoints in metres / radians) and writes
        a BEV trajectory plot to ``{eval_dir}/{step_tag}_bev.png``. Also reports ADE/FDE/
        yaw error in the figure title so you can read the metric off the file without
        opening it programmatically.

        Returns the output path or ``None`` if matplotlib is unavailable.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        def _to_np(t):
            if hasattr(t, "detach"):
                t = t.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            if hasattr(t, "numpy"):
                t = t.numpy()
            return np.asarray(t, dtype=np.float64)

        pred = _to_np(pred_action_denorm).reshape(-1, 3)
        gt = _to_np(gt_action_denorm).reshape(-1, 3)
        if pred.shape != gt.shape:
            raise ValueError(
                f"pred/gt action shape mismatch: pred={pred.shape}, gt={gt.shape}"
            )

        # BEV in driving convention: forward = up (+y axis), right = +x axis.
        # Anchor ego frame uses +x forward, +y left; we plot (-dy, dx) and prepend the
        # anchor (0, 0) so the curve starts at the ego origin.
        def _to_bev(action: np.ndarray):
            xs = np.concatenate([[0.0], -action[:, 1]])
            ys = np.concatenate([[0.0], action[:, 0]])
            yaws = np.concatenate([[0.0], action[:, 2]])
            return xs, ys, yaws

        pred_xs, pred_ys, pred_yaws = _to_bev(pred)
        gt_xs, gt_ys, gt_yaws = _to_bev(gt)

        # Metrics, in physical units (m / rad).
        step_l2 = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)  # [T]
        ade = float(step_l2.mean()) if step_l2.size > 0 else 0.0
        fde = float(step_l2[-1]) if step_l2.size > 0 else 0.0
        yaw_err = float(np.abs(_wrap_to_pi(pred[:, 2] - gt[:, 2])).mean()) if pred.size > 0 else 0.0

        # Two-tone palette per series: a lighter shade for the connecting line, a darker
        # shade for the markers and heading arrows. Makes the relationship between waypoints
        # and arrows visually clear without losing the pred/gt distinction.
        GT_DARK = "#1b7a32"      # markers + arrows
        GT_LINE = "#6dbc7e"      # connector line (lighter green)
        PRED_DARK = "#6a1b9a"    # markers + arrows
        PRED_LINE = "#b388d6"    # connector line (lighter purple)

        fig, ax = plt.subplots(figsize=(6, 7))

        # Connector lines (lighter shades).
        ax.plot(gt_xs, gt_ys, color=GT_LINE, linewidth=2.5, zorder=2)
        ax.plot(pred_xs, pred_ys, color=PRED_LINE, linewidth=2.5, linestyle="--", zorder=2)

        # Waypoint markers (darker shades, on top of the lines).
        ax.scatter(gt_xs, gt_ys, color=GT_DARK, marker="o", s=55,
                   zorder=5, label="ground truth", edgecolors="white", linewidths=0.8)
        ax.scatter(pred_xs, pred_ys, color=PRED_DARK, marker="X", s=75,
                   zorder=5, label="predicted", edgecolors="white", linewidths=0.8)

        # Heading arrows (darker shades, matching the markers).
        for x, y, yaw in zip(gt_xs, gt_ys, gt_yaws):
            ax.arrow(
                x, y, 0.5 * np.sin(yaw), 0.5 * np.cos(yaw),
                head_width=0.25, length_includes_head=True, color=GT_DARK, alpha=0.85, zorder=4,
            )
        for x, y, yaw in zip(pred_xs, pred_ys, pred_yaws):
            ax.arrow(
                x, y, 0.5 * np.sin(yaw), 0.5 * np.cos(yaw),
                head_width=0.25, length_includes_head=True, color=PRED_DARK, alpha=0.85, zorder=4,
            )

        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        # `adjustable="box"` makes matplotlib resize the axes box to honor the aspect
        # ratio, so the xlim/ylim we set below are honored exactly. With the default
        # `adjustable="datalim"` matplotlib would expand the data range instead, which
        # is fine for visibility but breaks "the viewport is what I asked for".
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("right (m)")
        ax.set_ylabel("forward (m)")
        ax.set_title(
            f"{step_tag}\nADE={ade:.2f} m   FDE={fde:.2f} m   yaw_err={yaw_err:.3f} rad"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Fixed viewport (in meters) for visual consistency across eval steps. Only
        # expand the bounds if any waypoint (or its heading-arrow tip) would fall
        # outside; an unusual sample with long trajectories or large lateral motion
        # is never clipped. Defaults assume nuScenes urban driving: ~40 m forward over
        # 4 s @ 10 m/s, ~10 m lateral on sharp turns.
        DEFAULT_X = (-15.0, 15.0)
        DEFAULT_Y = (-5.0, 40.0)
        ARROW_REACH = 0.75   # arrow length (0.5) + head_width (0.25)
        EDGE_PAD = 1.5       # breathing room past the arrow tip so nothing sits on the axis line

        all_x = np.concatenate([pred_xs, gt_xs])
        all_y = np.concatenate([pred_ys, gt_ys])
        data_x_lo = float(all_x.min()) - ARROW_REACH - EDGE_PAD
        data_x_hi = float(all_x.max()) + ARROW_REACH + EDGE_PAD
        data_y_lo = float(all_y.min()) - ARROW_REACH - EDGE_PAD
        data_y_hi = float(all_y.max()) + ARROW_REACH + EDGE_PAD

        x_lo = min(DEFAULT_X[0], data_x_lo)
        x_hi = max(DEFAULT_X[1], data_x_hi)
        y_lo = min(DEFAULT_Y[0], data_y_lo)
        y_hi = max(DEFAULT_Y[1], data_y_hi)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        fig.tight_layout()
        os.makedirs(eval_dir, exist_ok=True)
        out_path = os.path.join(eval_dir, f"{step_tag}_bev.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path


if __name__ == "__main__":
    # Quick standalone explorer for NuScenesVideoDataset. Renders video strips and BEV
    # waypoint plots for a few samples so you can eyeball that the trajectory is
    # forward-going, units look sane, and the image stream is the right scene.
    #
    # Run from anywhere (paths are resolved off __file__):
    #     python src/fastwam/datasets/nuscenes/nuscenes_video_dataset.py
    #     python src/fastwam/datasets/nuscenes/nuscenes_video_dataset.py --indices 0 5 20 --split val
    import argparse
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    parser = argparse.ArgumentParser(description="Inspect NuScenesVideoDataset samples.")
    parser.add_argument("--task", default="nuscenes_uncond_1cam224_1e-4")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "mini"],
        help="train/val auto-map to mini_train/mini_val when version=v1.0-mini; "
             "mini = all v1.0-mini scenes combined (no inner train/val split).",
    )
    parser.add_argument("--indices", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--out-dir", default="./tmp/nuscenes_inspect")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    config_dir = str(repo_root / "configs")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    ds = instantiate(cfg.data[args.split])
    print(f"len({args.split}) = {len(ds)}")

    norm_action = ds.processor.normalizer.normalizers["action"][ds._action_key]
    norm_state = ds.processor.normalizer.normalizers["state"][ds._state_key]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        have_mpl = True
    except ImportError:
        have_mpl = False
        print("(matplotlib not installed; skipping BEV plots)")

    for idx in args.indices:
        if idx >= len(ds):
            print(f"skip idx={idx} (>= len {len(ds)})")
            continue
        sample = ds[idx]
        print(f"\n=== sample {idx} ===")
        for k, v in sample.items():
            if hasattr(v, "shape"):
                line = f"  {k:14s} shape={tuple(v.shape)} dtype={v.dtype}"
                if v.dtype.is_floating_point:
                    line += f"  min={v.min().item():+.3f}  max={v.max().item():+.3f}"
                print(line)
            else:
                preview = repr(v)
                print(f"  {k:14s} {preview[:96]}")

        action_raw = norm_action.backward(sample["action"].clone()).cpu().numpy()
        proprio_raw = norm_state.backward(sample["proprio"].clone()).cpu().numpy()
        print(f"  action_raw (m, m, rad) first 3 steps:\n{action_raw[:3]}")
        print(
            f"  proprio_raw (vx, vy, yaw_rate, speed) step0: {proprio_raw[0]}, "
            f"speed range [{proprio_raw[:, 3].min():.2f}, {proprio_raw[:, 3].max():.2f}] m/s"
        )

        # Save horizontal strip of T_video frames
        video = sample["video"].float().clamp(-1, 1)
        frames = ((video + 1.0) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,C]
        strip = np.concatenate(list(frames), axis=1)
        Image.fromarray(strip).save(out_dir / f"sample_{idx:05d}_frames.png")

        if have_mpl:
            # BEV in driving convention: forward = up (+y axis), right = +x axis.
            # Anchor ego frame has +x forward and +y left, so we plot (-dy, dx).
            xs = np.concatenate([[0.0], -action_raw[:, 1]])
            ys = np.concatenate([[0.0], action_raw[:, 0]])
            yaws = np.concatenate([[0.0], action_raw[:, 2]])

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(xs, ys, marker="o", color="C0")
            for x, y, yaw in zip(xs, ys, yaws):
                # heading arrow: anchor yaw is +x_forward; in BEV (right, up) becomes (sin(yaw), cos(yaw)).
                ax.arrow(
                    x, y, 0.6 * np.sin(yaw), 0.6 * np.cos(yaw),
                    head_width=0.3, length_includes_head=True, color="C1", alpha=0.7,
                )
            ax.axhline(0, color="0.7", lw=0.5)
            ax.axvline(0, color="0.7", lw=0.5)
            ax.set_aspect("equal")
            ax.set_xlabel("right (m)")
            ax.set_ylabel("forward (m)")
            ax.set_title(f"sample {idx}: ego trajectory ({len(action_raw)} steps)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{idx:05d}_bev.png", dpi=120)
            plt.close(fig)

    print(f"\nWrote outputs to {out_dir.resolve()}")
