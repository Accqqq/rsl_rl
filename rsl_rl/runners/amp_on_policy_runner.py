# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""Keep the AMP_mjlab configuration API on the current runner implementation."""

from __future__ import annotations

from copy import deepcopy

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def prepare_amp_config(train_cfg: dict) -> dict:
    """Translate legacy top-level AMP parameters without mutating caller config."""
    cfg = deepcopy(train_cfg)
    algorithm = cfg["algorithm"]
    algorithm["class_name"] = "rsl_rl.algorithms:AMPPPO"
    keys = (
        "amp_reward_coef",
        "amp_motion_files",
        "amp_task_reward_lerp",
        "amp_discr_hidden_dims",
        "amp_replay_buffer_size",
        "amp_grad_penalty",
        "amp_learning_rate",
        "amp_lr_schedule",
        "amp_num_updates",
        "amp_batch_size",
        "amp_max_grad_norm",
        "amp_body_names",
        "amp_anchor_name",
        "amp_all_body_names",
        "amp_frame_dt",
        "amp_observation_group",
    )
    for key in keys:
        if key in cfg:
            algorithm.setdefault(key, cfg.pop(key))
    if "min_normalized_std" in cfg:
        algorithm.setdefault("min_std", cfg.pop("min_normalized_std"))
    # The loader now vectorizes complete clips; it no longer duplicates randomly preloaded transitions.
    cfg.pop("amp_num_preload_transitions", None)
    return cfg


class AmpOnPolicyRunner(OnPolicyRunner):
    """Run AMP using the current upstream on-policy learning loop."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Translate legacy AMP parameters before constructing the upstream runner."""
        super().__init__(env, prepare_amp_config(train_cfg), log_dir, device)
