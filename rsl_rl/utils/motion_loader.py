# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The Legged Lab and TienKung-Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent AMP features from AMP_mjlab motion archives."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Protocol


class AMPData(Protocol):
    """Minimal expert sampler interface accepted by AMPPPO."""

    observation_dim: int

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample continuous expert feature pairs."""
        ...


def _quat_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(quaternion).all() or (quaternion.norm(dim=-1) < 1e-8).any():
        raise ValueError("AMP body rotations must contain finite nonzero wxyz quaternions")
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def compute_amp_features(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    linear: torch.Tensor,
    angular: torch.Tensor,
    body_indices: Sequence[int] | torch.Tensor | slice,
    anchor_index: int,
) -> torch.Tensor:
    """Build identical term-major features for archived frames and live states.

    Inputs have shape [batch, all_bodies, xyz/wxyz]. The batch can represent
    expert frames or parallel environments. Velocities use each body's frame.
    """
    rotation = _quat_matrix(quaternion)
    inverse_anchor = rotation[:, anchor_index].transpose(-1, -2).unsqueeze(1)
    local_position = (
        inverse_anchor @ (position[:, body_indices] - position[:, anchor_index, None]).unsqueeze(-1)
    ).squeeze(-1)
    local_rotation = inverse_anchor @ rotation[:, body_indices]
    inverse_body = rotation[:, body_indices].transpose(-1, -2)
    return torch.cat(
        (
            local_position.flatten(1),
            local_rotation[..., :2].flatten(1),
            (inverse_body @ linear[:, body_indices].unsqueeze(-1)).squeeze(-1).flatten(1),
            (inverse_body @ angular[:, body_indices].unsqueeze(-1)).squeeze(-1).flatten(1),
        ),
        dim=-1,
    )


class AMPLoader:
    """Cache clip-local feature pairs without depending on a simulator package.

    Features keep the original order: all anchor-relative body positions, all
    anchor-relative 6D rotations, all body-local linear and angular velocities.
    Archives may instead contain precomputed [frames, features] AMP observations.
    """

    def __init__(
        self,
        motion_file: str | Path,
        body_names: Sequence[str] = (),
        anchor_name: str = "",
        all_body_names: Sequence[str] = (),
        device: str | torch.device = "cpu",
        frame_dt: float | None = None,
    ) -> None:
        """Load NPZ clips and check their schema and temporal sampling interval."""
        root = Path(motion_file)
        paths = sorted(root.rglob("*.npz")) if root.is_dir() else [root]
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"No AMP motion archives found at {root}")
        self.motion_names = [path.stem for path in paths]
        self.clips = []
        self.strides = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Invalid motion FPS in {path}")
                frame_stride = fps * frame_dt if frame_dt is not None else 1.0
                stride = round(frame_stride)
                if stride < 1 or not np.isclose(stride, frame_stride, atol=1e-5):
                    raise ValueError(f"Resample {path} so its frame interval matches control dt={frame_dt}")
                if "amp_observations" in archive:
                    features = torch.as_tensor(archive["amp_observations"], dtype=torch.float32, device=device)
                else:
                    names = list(archive["body_names"].astype(str)) if "body_names" in archive else list(all_body_names)
                    if not names or not body_names or not anchor_name:
                        raise ValueError("Raw motion archives need body names, selected AMP bodies, and an anchor")
                    indices = [names.index(name) for name in body_names]
                    anchor = names.index(anchor_name)
                    position = torch.as_tensor(archive["body_pos_w"], dtype=torch.float32, device=device)
                    quaternion = torch.as_tensor(archive["body_quat_w"], dtype=torch.float32, device=device)
                    if position.shape[1] != len(names) or quaternion.shape[:2] != position.shape[:2]:
                        raise ValueError(f"Motion body schema does not match robot body names in {path}")
                    linear = torch.as_tensor(archive["body_lin_vel_w"], dtype=torch.float32, device=device)
                    angular = torch.as_tensor(archive["body_ang_vel_w"], dtype=torch.float32, device=device)
                    features = compute_amp_features(position, quaternion, linear, angular, indices, anchor)
                if features.ndim != 2 or features.shape[0] <= stride or not torch.isfinite(features).all():
                    raise ValueError(f"AMP motion must contain finite features and at least two usable frames: {path}")
                if self.clips and features.shape[1] != self.clips[0].shape[1]:
                    raise ValueError("All AMP clips must have the same feature dimension")
                self.clips.append(features)
                self.strides.append(stride)
        self.observation_dim = self.clips[0].shape[1]
        self.device = device

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a clip uniformly and a valid pair within that clip."""
        # Uniform clip sampling preserves minority motions; frames never cross clip boundaries.
        clip_ids = torch.randint(len(self.clips), (batch_size,), device=self.device)
        state = torch.empty(batch_size, self.observation_dim, device=self.device)
        next_state = torch.empty_like(state)
        for index, (clip, stride) in enumerate(zip(self.clips, self.strides)):
            mask = clip_ids == index
            count = int(mask.sum())
            if count:
                frames = torch.randint(clip.shape[0] - stride, (count,), device=self.device)
                state[mask], next_state[mask] = clip[frames], clip[frames + stride]
        return state, next_state

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Expose the original AMP expert mini-batch API."""
        for _ in range(num_mini_batch):
            yield self.sample(mini_batch_size)
