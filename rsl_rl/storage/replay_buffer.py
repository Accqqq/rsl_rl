# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The Legged Lab and TienKung-Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded transition replay for AMP discriminator training."""

from __future__ import annotations

import torch
from collections.abc import Generator


class ReplayBuffer:
    """Keep bounded, paired feature frames with uniform transition sampling."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str | torch.device = "cpu") -> None:
        """Allocate fixed-capacity storage on the learning device."""
        if obs_dim < 1 or buffer_size < 1:
            raise ValueError("Replay dimensions and capacity must be positive")
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros_like(self.states)
        self.buffer_size = buffer_size
        self.step = 0
        self.num_samples = 0

    @torch.no_grad()
    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Copy paired frames, wrapping and retaining the latest capacity-sized batch."""
        if states.shape != next_states.shape or states.ndim != 2 or states.shape[1] != self.states.shape[1]:
            raise ValueError("AMP replay expects matching [batch, feature] state pairs")
        if states.shape[0] >= self.buffer_size:
            states, next_states = states[-self.buffer_size :], next_states[-self.buffer_size :]
        count = states.shape[0]
        first = min(count, self.buffer_size - self.step)
        self.states[self.step : self.step + first].copy_(states[:first])
        self.next_states[self.step : self.step + first].copy_(next_states[:first])
        if first < count:
            self.states[: count - first].copy_(states[first:])
            self.next_states[: count - first].copy_(next_states[first:])
        self.step = (self.step + count) % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + count)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample transitions with replacement."""
        if not self.num_samples:
            raise RuntimeError("Cannot sample an empty AMP replay buffer")
        indices = torch.randint(self.num_samples, (batch_size,), device=self.states.device)
        return self.states[indices], self.next_states[indices]

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Expose the original AMP replay mini-batch API."""
        for _ in range(num_mini_batch):
            yield self.sample(mini_batch_size)
