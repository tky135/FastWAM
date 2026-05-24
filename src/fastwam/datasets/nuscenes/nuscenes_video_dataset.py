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
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir() or "."
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )

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
        if version.endswith("mini"):
            key = "mini_train" if split == "train" else "mini_val"
        else:
            key = "train" if split == "train" else "val"
        if key not in all_splits:
            raise ValueError(
                f"Split '{key}' not found in nuscenes splits. Available: {list(all_splits)}"
            )
        wanted = set(all_splits[key])
        present = {s["name"]: s["token"] for s in self.nusc.scene}
        scene_names = sorted(wanted & set(present.keys()))
        if not scene_names:
            raise RuntimeError(
                f"No scenes from split '{key}' present in dataroot '{self.dataroot}'."
            )
        return scene_names

    def _build_sample_index(self, scene_names: list[str]) -> list[str]:
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
            if max_anchor <= 0:
                continue
            for i in range(0, max_anchor, self.global_sample_stride):
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
        translations: np.ndarray,
        rotations: np.ndarray,
        samples: list[dict],
    ) -> torch.Tensor:
        T = translations.shape[0]
        timestamps_us = np.asarray([s["timestamp"] for s in samples], dtype=np.float64)
        dt = (timestamps_us[1:] - timestamps_us[:-1]) * 1e-6  # [T-1]
        dt = np.where(dt <= 1e-6, 1e-3, dt)  # guard against zero

        yaws = self._yaw_from_quat(rotations)

        v_world = np.zeros((T, 2), dtype=np.float64)
        v_world[:-1, 0] = (translations[1:, 0] - translations[:-1, 0]) / dt
        v_world[:-1, 1] = (translations[1:, 1] - translations[:-1, 1]) / dt
        v_world[-1] = v_world[-2]  # backward diff for last step

        cos_y = np.cos(-yaws)
        sin_y = np.sin(-yaws)
        v_x = cos_y * v_world[:, 0] - sin_y * v_world[:, 1]
        v_y = sin_y * v_world[:, 0] + cos_y * v_world[:, 1]
        speed = np.sqrt(v_x * v_x + v_y * v_y)

        yaw_rate = np.zeros(T, dtype=np.float64)
        yaw_rate[:-1] = _wrap_to_pi(yaws[1:] - yaws[:-1]) / dt
        yaw_rate[-1] = yaw_rate[-2]

        proprio = np.stack([v_x, v_y, yaw_rate, speed], axis=-1).astype(np.float32)
        return torch.from_numpy(proprio)

    def _build_raw_sample(self, idx: int) -> dict:
        anchor_token = self.sample_tokens[idx]
        samples = self._walk_forward_keyframes(anchor_token, self.num_frames)
        imgs = torch.stack(
            [self._load_front_image(s) for s in samples], dim=0
        )  # [T, 3, H, W] uint8
        translations = np.stack([self._get_ego_pose(s)[0] for s in samples], axis=0)
        rotations = np.stack([self._get_ego_pose(s)[1] for s in samples], axis=0)
        actions = self._compute_waypoints(translations, rotations)  # [T-1, 3]
        proprio = self._compute_proprio(translations, rotations, samples)  # [T, 4]

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
        from tqdm import tqdm

        action_list: list[torch.Tensor] = []
        state_list: list[torch.Tensor] = []
        for i in tqdm(range(len(self)), desc="nuScenes stats pass"):
            raw = self._build_raw_sample(i)
            transformed = processor.action_state_transform(
                {
                    "action": {k: v.clone() for k, v in raw["action"].items()},
                    "state": {k: v.clone() for k, v in raw["state"].items()},
                }
            )
            action_list.append(transformed["action"][self._action_key])
            state_list.append(transformed["state"][self._state_key])

        action = torch.stack(action_list, dim=0)  # [N, T-1, A]
        state = torch.stack(state_list, dim=0)  # [N, T, S]

        def _per_key_stats(data: torch.Tensor) -> dict:
            data = data.float()
            stepwise_min = data.amin(dim=0)
            stepwise_max = data.amax(dim=0)
            stepwise_mean = data.mean(dim=0)
            stepwise_std = data.std(dim=0, unbiased=False)
            stepwise_q01 = torch.quantile(data, 0.01, dim=0)
            stepwise_q99 = torch.quantile(data, 0.99, dim=0)
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
            "video": video,
            "action": sample["action"],
            "proprio": sample["proprio"][:-1, :],
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._get(idx)
        except Exception as exc:
            print(f"Error processing nuScenes sample idx {idx}: {exc}")
            print(traceback.format_exc())
            return self._get(int(np.random.randint(len(self))))


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
    parser.add_argument("--split", default="train", choices=["train", "val"])
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
