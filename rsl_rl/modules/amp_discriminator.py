# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The Legged Lab and TienKung-Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Least-squares AMP discriminator, ported from AMP_mjlab."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from torch import nn

from rsl_rl.modules.normalization import EmpiricalNormalization


class AMPStateNormalizer(EmpiricalNormalization):
    """Torch-native running moments with the legacy AMP clipping convention."""

    _mean: torch.Tensor
    _var: torch.Tensor

    def __init__(self, shape: int, clip: float = 10.0, eps: float = 1e-4) -> None:
        """Initialize checkpointable moments and the AMP clipping range."""
        super().__init__(shape, eps=eps)
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize without updating moments during reward calculation."""
        return ((x - self._mean) / torch.sqrt(self._var + self.eps)).clamp(-self.clip, self.clip)

    def normalize_torch(self, x: torch.Tensor, device: str | torch.device | None = None) -> torch.Tensor:
        """Compatibility entry point used by the original AMP discriminator."""
        return self(x.to(device) if device is not None else x)


class Discriminator(nn.Module):
    """Classify consecutive motion features; expert scores target +1, policy -1."""

    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float,
        hidden_layer_sizes: Sequence[int],
        device: str | torch.device = "cpu",
        task_reward_lerp: float = 0.0,
    ) -> None:
        """Initialize a discriminator over two consecutive feature frames."""
        super().__init__()
        if input_dim < 2 or input_dim % 2 or not hidden_layer_sizes:
            raise ValueError("AMP requires a pair of equally sized feature vectors and nonempty hidden layers")
        if not 0 <= task_reward_lerp <= 1 or amp_reward_coef < 0:
            raise ValueError("task_reward_lerp must be in [0, 1] and amp_reward_coef nonnegative")
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp
        layers = []
        previous = input_dim
        for size in hidden_layer_sizes:
            layers.extend((nn.Linear(previous, size), nn.ReLU()))
            previous = size
        self.trunk = nn.Sequential(*layers)
        self.amp_linear = nn.Linear(previous, 1)
        self.to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return an unbounded motion classification score."""
        return self.amp_linear(self.trunk(features))

    def compute_grad_pen(
        self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0
    ) -> torch.Tensor:
        """Penalize the squared input-gradient norm on expert transitions."""
        features = torch.cat((expert_state, expert_next_state), dim=-1).detach().requires_grad_(True)
        score = self(features)
        gradient = torch.autograd.grad(score.sum(), features, create_graph=True)[0]
        return lambda_ * gradient.square().sum(dim=-1).mean()

    @torch.no_grad()
    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer: AMPStateNormalizer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the original AMP/task reward blend and the discriminator score."""
        if normalizer is not None:
            normalize = normalizer.normalize_torch if hasattr(normalizer, "normalize_torch") else normalizer
            state = normalize(state)
            next_state = normalize(next_state)
        score = self(torch.cat((state, next_state), dim=-1))
        style = self.amp_reward_coef * (1 - 0.25 * (score - 1).square()).clamp_min(0).squeeze(-1)
        reward = (1 - self.task_reward_lerp) * style + self.task_reward_lerp * task_reward
        return reward, score
