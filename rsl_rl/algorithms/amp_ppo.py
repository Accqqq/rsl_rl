# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The Legged Lab and TienKung-Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP on the current TensorDict/actor/critic PPO interface."""

from __future__ import annotations

import torch
from math import isfinite
from tensordict import TensorDict
from torch import nn
from typing import Any, Literal

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.amp_discriminator import AMPStateNormalizer, Discriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.utils import reduce_gradients_in_buckets
from rsl_rl.utils.motion_loader import AMPData, AMPLoader


class AMPPPO(PPO):
    """Extend current PPO with motion-style rewards and discriminator updates.

    Environments provide a single AMP feature frame in obs[amp_observation_group].
    With automatic resets, done transitions are excluded unless extras supplies
    amp_terminal_observations [num_envs, features] from BEFORE the reset.
    This prevents learning the teleport from a fall into a new episode.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        amp_observation_group: str = "amp",
        amp_reward_coef: float = 0.1,
        amp_task_reward_lerp: float = 0.75,
        amp_discr_hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 256),
        amp_replay_buffer_size: int = 100000,
        amp_grad_penalty: float = 10.0,
        amp_learning_rate: float = 1e-3,
        amp_lr_schedule: Literal["fixed", "ppo"] = "fixed",
        amp_num_updates: int = 20,
        amp_batch_size: int = 512,
        amp_max_grad_norm: float | None = None,
        amp_motion_files: str = "",
        amp_body_names: tuple[str, ...] = (),
        amp_anchor_name: str = "",
        amp_all_body_names: tuple[str, ...] = (),
        amp_frame_dt: float | None = None,
        amp_data: AMPData | None = None,
        min_std: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Keep upstream PPO settings and add the AMP feature, data, and optimizer settings."""
        super().__init__(actor, critic, storage, **kwargs)
        if not isfinite(amp_learning_rate) or amp_learning_rate <= 0:
            raise ValueError("amp_learning_rate must be finite and positive")
        if amp_lr_schedule not in ("fixed", "ppo"):
            raise ValueError("amp_lr_schedule must be 'fixed' or 'ppo'")
        for name, value in (("amp_num_updates", amp_num_updates), ("amp_batch_size", amp_batch_size)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if amp_max_grad_norm is not None and (not isfinite(amp_max_grad_norm) or amp_max_grad_norm <= 0):
            raise ValueError("amp_max_grad_norm must be None or finite and positive")
        self.amp_learning_rate = amp_learning_rate
        self.amp_lr_schedule = amp_lr_schedule
        self.amp_num_updates = amp_num_updates
        self.amp_batch_size = amp_batch_size
        self.amp_max_grad_norm = amp_max_grad_norm
        if amp_observation_group not in storage.observations:
            raise ValueError(f"Missing AMP observation group {amp_observation_group!r}")
        sample = storage.observations[amp_observation_group]
        if sample.ndim != 3:
            raise ValueError("AMP observations must be single feature frames [num_envs, features]")
        self.amp_observation_group = amp_observation_group
        self.amp_observation_dim = sample.shape[-1]
        self.discriminator = Discriminator(
            2 * self.amp_observation_dim,
            amp_reward_coef,
            amp_discr_hidden_dims,
            self.device,
            amp_task_reward_lerp,
        )
        self.amp_normalizer = AMPStateNormalizer(self.amp_observation_dim).to(self.device)
        self.amp_storage = ReplayBuffer(self.amp_observation_dim, amp_replay_buffer_size, self.device)
        self.amp_optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-1},
            ],
            lr=self.learning_rate if self.amp_lr_schedule == "ppo" else self.amp_learning_rate,
        )
        self.amp_grad_penalty = amp_grad_penalty
        if amp_grad_penalty < 0:
            raise ValueError("amp_grad_penalty must be nonnegative")
        self.amp_data: AMPData | None = amp_data
        self._amp_motion_config: dict[str, Any] = dict(
            motion_file=amp_motion_files,
            body_names=amp_body_names,
            anchor_name=amp_anchor_name,
            all_body_names=amp_all_body_names,
            device=self.device,
            frame_dt=amp_frame_dt,
        )
        self._amp_state: torch.Tensor | None = None
        self.min_std: torch.Tensor | None = None
        if min_std is not None:
            minimum = torch.as_tensor(min_std, device=self.device)
            if minimum.numel() not in (1, storage.actions.shape[-1]) or (minimum < 0).any():
                raise ValueError("min_std must contain one or num_actions nonnegative values")
            distribution = self._raw_actor.distribution
            scalar_std = getattr(distribution, "std_param", None)
            log_std = getattr(distribution, "log_std_param", None)
            if not isinstance(scalar_std, torch.Tensor) and not isinstance(log_std, torch.Tensor):
                raise ValueError("AMP min_std requires a state-independent Gaussian distribution")
            self.min_std = minimum

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Remember the current feature frame before the environment advances."""
        self._amp_state = obs[self.amp_observation_group].detach().clone()
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, Any]
    ) -> None:
        """Mix style rewards and store only physically continuous state pairs."""
        if self._amp_state is None:
            raise RuntimeError("Call act before processing an AMP transition")
        next_state = obs[self.amp_observation_group].detach().clone()
        done = dones.bool()
        valid = ~done
        terminal = extras.get("amp_terminal_observations")
        if terminal is not None:
            if terminal.shape != next_state.shape:
                raise ValueError("amp_terminal_observations must match the AMP observation shape")
            available = extras.get("amp_terminal_mask", torch.ones_like(done))
            terminal_ids = done & available.bool()
            next_state[terminal_ids] = terminal[terminal_ids].to(self.device)
            valid |= terminal_ids
        if not torch.isfinite(next_state[valid]).all():
            raise ValueError("Non-finite AMP transition features")
        mixed_reward, score = self.discriminator.predict_amp_reward(
            self._amp_state,
            next_state,
            rewards,
            self.amp_normalizer,
        )
        # Preserve the task's terminal penalty when its terminal features are unavailable.
        mixed_reward = torch.where(valid, mixed_reward, rewards)
        self.amp_storage.insert(self._amp_state[valid], next_state[valid])
        log = extras.setdefault("episode" if "episode" in extras else "log", {})
        log["AMP/task_reward"] = rewards.detach().mean()
        log["AMP/combined_reward"] = mixed_reward.detach().mean()
        log["AMP/valid_transition_fraction"] = valid.float().mean()
        log["AMP/policy_score"] = score[valid].mean() if valid.any() else score.new_zeros(())
        super().process_env_step(obs, mixed_reward, dones, extras)
        self._amp_state = None

    def _expert_data(self) -> AMPData:
        if self.amp_data is None:
            if not self._amp_motion_config["motion_file"]:
                raise ValueError("AMP training requires amp_motion_files or an amp_data sampler")
            self.amp_data = AMPLoader(**self._amp_motion_config)
        if self.amp_data.observation_dim != self.amp_observation_dim:
            raise ValueError("Expert features and environment AMP observations have different dimensions")
        return self.amp_data

    def _set_amp_learning_rate(self) -> None:
        """Use the independent AMP rate unless following PPO was explicitly selected."""
        rate = self.learning_rate if self.amp_lr_schedule == "ppo" else self.amp_learning_rate
        for group in self.amp_optimizer.param_groups:
            group["lr"] = rate

    def update(self) -> dict[str, float]:
        """Run upstream PPO, then train the least-squares motion discriminator."""
        # Upstream retains ownership of PPO, RND, symmetry, mixed precision, and model normalization.
        losses = super().update()
        self._clamp_std()
        ready = torch.tensor(int(self.amp_storage.num_samples > 0), device=self.device)
        if self.is_multi_gpu:
            # Every rank must enter the same discriminator/normalizer collectives.
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        if not ready:
            losses["amp_skipped"] = 1.0
            return losses
        expert = self._expert_data()
        totals = {
            key: 0.0 for key in ("amp_discriminator", "amp_gradient_penalty", "amp_policy_score", "amp_expert_score")
        }
        self._set_amp_learning_rate()
        for _ in range(self.amp_num_updates):
            state, next_state = self.amp_storage.sample(self.amp_batch_size)
            expert_state, expert_next = expert.sample(self.amp_batch_size)
            expert_state, expert_next = expert_state.to(self.device), expert_next.to(self.device)
            normalized = [self.amp_normalizer(x) for x in (state, next_state, expert_state, expert_next)]
            policy_score = self.discriminator(torch.cat(normalized[:2], dim=-1))
            expert_score = self.discriminator(torch.cat(normalized[2:], dim=-1))
            classification = 0.5 * ((policy_score + 1).square().mean() + (expert_score - 1).square().mean())
            gradient_penalty = self.discriminator.compute_grad_pen(*normalized[2:], lambda_=self.amp_grad_penalty)
            loss = classification + gradient_penalty
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite AMP discriminator loss")
            self.amp_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                reduce_gradients_in_buckets(
                    self.discriminator.parameters(), self.gpu_world_size, self.grad_reduce_bucket_mb
                )
            # Infinity checks gradient finiteness without clipping when AMP clipping is disabled.
            norm_limit = self.amp_max_grad_norm if self.amp_max_grad_norm is not None else float("inf")
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), norm_limit, error_if_nonfinite=True)
            self.amp_optimizer.step()
            with torch.no_grad():
                self.amp_normalizer.update(torch.cat((state, next_state, expert_state, expert_next)))
            for key, value in zip(totals, (classification, gradient_penalty, policy_score.mean(), expert_score.mean())):
                totals[key] += float(value.detach()) / self.amp_num_updates
        return {**losses, **totals}

    @torch.no_grad()
    def _clamp_std(self) -> None:
        if self.min_std is None:
            return
        distribution = self._raw_actor.distribution
        scalar_std = getattr(distribution, "std_param", None)
        log_std = getattr(distribution, "log_std_param", None)
        if isinstance(scalar_std, torch.Tensor):
            scalar_std.copy_(torch.maximum(scalar_std, self.min_std))
        elif isinstance(log_std, torch.Tensor):
            minimum = self.min_std.clamp_min(torch.finfo(torch.float32).tiny).log()
            log_std.copy_(torch.maximum(log_std, minimum))

    def train_mode(self) -> None:
        """Enable training for PPO and AMP running statistics."""
        super().train_mode()
        self.discriminator.train()
        self.amp_normalizer.train()

    def eval_mode(self) -> None:
        """Freeze running statistics during evaluation."""
        super().eval_mode()
        self.discriminator.eval()
        self.amp_normalizer.eval()

    def save(self) -> dict:
        """Save learned AMP state together with the upstream PPO checkpoint."""
        return {
            **super().save(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "amp_normalizer_state_dict": self.amp_normalizer.state_dict(),
            "amp_optimizer_state_dict": self.amp_optimizer.state_dict(),
            "amp_observation_dim": self.amp_observation_dim,
            # Audit metadata: cadence, clipping, and schedule are selected by the current run config.
            "amp_training_config": {
                "amp_learning_rate": self.amp_optimizer.param_groups[0]["lr"],
                "amp_lr_schedule": self.amp_lr_schedule,
                "amp_num_updates": self.amp_num_updates,
                "amp_batch_size": self.amp_batch_size,
                "amp_max_grad_norm": self.amp_max_grad_norm,
            },
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore AMP training state, or allow actor-only loading for inference."""
        if load_cfg is None or load_cfg.get("amp", False):
            if loaded_dict.get("amp_observation_dim") != self.amp_observation_dim:
                raise ValueError(
                    "Full AMP resume requires a matching AMP checkpoint; use actor-only loading for inference"
                )
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"], strict=strict)
            if load_cfg is None or load_cfg.get("optimizer", False):
                self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
                # Match upstream PPO resume semantics: restore optimizer moments and its saved rate.
                self.amp_learning_rate = self.amp_optimizer.param_groups[0]["lr"]
        self._amp_state = None
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self._set_amp_learning_rate()
        return load_iteration

    def broadcast_parameters(self) -> None:
        """Synchronize both PPO and AMP modules at distributed startup."""
        super().broadcast_parameters()
        for module in (self.discriminator, self.amp_normalizer):
            for tensor in module.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)
