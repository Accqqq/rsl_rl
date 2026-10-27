# Copyright (c) 2026, AMP_mjlab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral checks for the AMP extension and current PPO API."""

from __future__ import annotations

import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from tensordict import TensorDict
from types import SimpleNamespace
from typing import Any

import pytest

from rsl_rl.algorithms import AMPPPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.amp_discriminator import Discriminator
from rsl_rl.runners.amp_on_policy_runner import prepare_amp_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.utils.motion_loader import AMPLoader, compute_amp_features


def build_amp(**amp_overrides: Any) -> tuple[AMPPPO, TensorDict]:
    """Construct a small real PPO/AMP setup with deterministic expert data."""
    obs = TensorDict({"policy": torch.randn(4, 8), "amp": torch.randn(4, 6)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(obs, groups, "actor", 3, hidden_dims=[16], distribution_cfg={"class_name": "GaussianDistribution"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", 4, 4, obs, [3])
    expert = SimpleNamespace(observation_dim=6, sample=lambda n: (torch.ones(n, 6), torch.ones(n, 6) + 0.1))
    amp_cfg = {
        "amp_num_updates": 4,
        "amp_batch_size": 8,
        **amp_overrides,
    }
    alg = AMPPPO(
        actor,
        critic,
        storage,
        amp_data=expert,
        amp_discr_hidden_dims=[16],
        num_learning_epochs=2,
        num_mini_batches=2,
        amp_replay_buffer_size=32,
        min_std=[0.2] * 3,
        **amp_cfg,
    )
    return alg, obs


def collect_rollout(alg: AMPPPO, obs: TensorDict) -> TensorDict:
    """Collect a real PPO rollout with continuous feature pairs and computed returns."""
    with torch.inference_mode():
        for _ in range(alg.storage.num_transitions_per_env):
            alg.act(obs)
            next_obs = obs.clone()
            next_obs["amp"] += 0.1
            alg.process_env_step(next_obs, torch.rand(alg.storage.num_envs), torch.zeros(alg.storage.num_envs), {})
            obs = next_obs
        alg.compute_returns(obs)
    return obs


def test_amp_reward_matches_original_formula_and_preserves_mode() -> None:
    """Keep the original least-squares reward blend, including negative task rewards."""
    disc = Discriminator(4, 2.0, [4], task_reward_lerp=0.75)
    for parameter in disc.parameters():
        parameter.data.zero_()
    disc.amp_linear.bias.data.fill_(1)
    disc.eval()
    reward, score = disc.predict_amp_reward(torch.zeros(2, 2), torch.zeros(2, 2), torch.tensor([4.0, -4.0]))
    assert torch.allclose(reward, torch.tensor([3.5, -2.5]))
    assert torch.equal(score, torch.ones(2, 1))
    assert not disc.training


def test_replay_wraps_large_batches_and_keeps_state_pairs() -> None:
    """Keep both halves of a transition paired across capacity overflow."""
    replay = ReplayBuffer(1, 5)
    replay.insert(torch.arange(3.0)[:, None], torch.arange(3.0)[:, None] + 10)
    states = torch.arange(8.0)[:, None]
    replay.insert(states, states + 10)
    states.fill_(-1)
    assert sorted(replay.states.flatten().tolist()) == [3, 4, 5, 6, 7]
    current, following = replay.sample(64)
    assert torch.equal(following, current + 10)


@pytest.mark.parametrize("terminal_available", [False, True])
def test_auto_reset_frames_are_never_used_as_terminal_motion(terminal_available: bool) -> None:
    """Reject reset teleports, but accept real terminal frames when supplied."""
    alg, obs = build_amp()
    previous = obs["amp"].clone()
    following = obs.clone()
    following["amp"] += 100
    done = torch.tensor([1, 0, 1, 0])
    extras = {}
    if terminal_available:
        extras["amp_terminal_observations"] = previous + 0.1
    with torch.inference_mode():
        alg.act(obs)
        alg.process_env_step(following, torch.full((4,), -4.0), done, extras)
    expected_count = 4 if terminal_available else 2
    assert alg.amp_storage.num_samples == expected_count
    if terminal_available:
        assert torch.allclose(alg.amp_storage.next_states[:4][done.bool()], previous[done.bool()] + 0.1)
    else:
        assert torch.allclose(alg.amp_storage.states[:2], previous[~done.bool()])
        assert torch.equal(alg.storage.rewards[0, done.bool(), 0], torch.full((2,), -4.0))


def test_current_ppo_and_discriminator_both_learn_and_resume() -> None:
    """Check actual parameter updates and restoration of all learned AMP state."""
    torch.manual_seed(7)
    alg, obs = build_amp()
    actor_before = deepcopy(alg._raw_actor.state_dict())
    disc_before = deepcopy(alg.discriminator.state_dict())
    collect_rollout(alg, obs)
    losses = alg.update()
    assert all(np.isfinite(value) for value in losses.values())
    assert "amp_discriminator" in losses and "surrogate" in losses
    assert any(not torch.equal(actor_before[key], value) for key, value in alg._raw_actor.state_dict().items())
    assert any(not torch.equal(disc_before[key], value) for key, value in alg.discriminator.state_dict().items())
    assert alg.amp_normalizer.count > 0
    state = deepcopy(alg.save())
    restored, _ = build_amp()
    assert restored.load(state, None, True)
    for key, value in alg.discriminator.state_dict().items():
        assert torch.equal(value, restored.discriminator.state_dict()[key])
    assert torch.equal(alg.amp_normalizer.mean, restored.amp_normalizer.mean)
    assert restored.amp_optimizer.state_dict()["state"]
    plain = {
        key: value for key, value in state.items() if not key.startswith("amp_") and key != "discriminator_state_dict"
    }
    assert not restored.load(plain, {"actor": True}, True)
    with pytest.raises(ValueError, match="Full AMP resume"):
        restored.load(plain, None, True)


def test_loader_pairs_stay_inside_each_clip_and_match_control_time(tmp_path: Path) -> None:
    """Never pair different clips or silently use a different control interval."""
    for index in range(2):
        features = np.arange(18, dtype=np.float32).reshape(6, 3) + index * 1000
        np.savez(tmp_path / f"{index}.npz", fps=50, amp_observations=features)
    loader = AMPLoader(tmp_path, frame_dt=0.02)
    state, following = loader.sample(100)
    assert torch.equal(following, state + 3)
    with pytest.raises(ValueError, match="Resample"):
        AMPLoader(tmp_path, frame_dt=1 / 30)


def test_raw_motion_features_use_anchor_positions_and_body_local_velocities(tmp_path: Path) -> None:
    """Verify feature order against a hand-computed raw motion frame."""
    position = np.zeros((3, 2, 3), np.float32)
    position[:, 1, 0] = 2
    quaternion = np.zeros((3, 2, 4), np.float32)
    quaternion[..., 0] = 1
    linear = np.ones_like(position)
    np.savez(
        tmp_path / "raw.npz",
        fps=50,
        body_names=np.array(["anchor", "foot"]),
        body_pos_w=position,
        body_quat_w=quaternion,
        body_lin_vel_w=linear,
        body_ang_vel_w=linear * 2,
    )
    loader = AMPLoader(tmp_path, ["foot"], "anchor", frame_dt=0.02)
    state, _ = loader.sample(1)
    expected = torch.tensor([[2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.float32)
    assert torch.equal(state, expected)


def test_legacy_amp_parameters_are_translated_without_mutating_input() -> None:
    """Keep the user-facing AMP configuration while using the new algorithm API."""
    cfg = {"algorithm": {"class_name": "AMPPPO"}, "amp_reward_coef": 0.1, "min_normalized_std": [0.05]}
    transformed = prepare_amp_config(cfg)
    assert transformed["algorithm"]["amp_reward_coef"] == pytest.approx(0.1)
    assert transformed["algorithm"]["min_std"] == [0.05]
    assert cfg["algorithm"] == {"class_name": "AMPPPO"}


def test_features_distinguish_anchor_and_body_coordinate_frames() -> None:
    """Check a 90-degree anchor rotation against hand-computed features."""
    half = 2**-0.5
    positions = torch.tensor([[[1.0, 2.0, 0.0], [1.0, 4.0, 0.0]]])
    rotations = torch.tensor([[[half, 0.0, 0.0, half], [1.0, 0.0, 0.0, 0.0]]])
    linear = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    features = compute_amp_features(positions, rotations, linear, linear * 2, [1], 0)
    expected = torch.tensor([[2, 0, 0, 0, 1, -1, 0, 0, 0, 1, 0, 0, 2, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(features, expected, atol=1e-6, rtol=1e-6)


def test_discriminator_update_count_and_batch_size_are_independent_of_ppo() -> None:
    """Actually run three discriminator steps of five samples while PPO runs four batches."""
    alg, obs = build_amp(amp_num_updates=3, amp_batch_size=5)
    requested_batches = []

    def sample_expert(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        requested_batches.append(batch_size)
        return torch.ones(batch_size, 6), torch.ones(batch_size, 6) + 0.1

    alg.amp_data = SimpleNamespace(observation_dim=6, sample=sample_expert)
    collect_rollout(alg, obs)
    alg.update()
    assert alg.num_learning_epochs * alg.num_mini_batches == 4
    assert requested_batches == [5, 5, 5]
    for state in alg.amp_optimizer.state.values():
        assert int(state["step"]) == 3
    # Both raw current/next policy and expert frames contribute to the moments.
    assert int(alg.amp_normalizer.count) == 3 * 5 * 4


@pytest.mark.parametrize("schedule, expected_rate", [("fixed", 3e-4), ("ppo", 9e-3)])
def test_discriminator_learning_rate_follows_ppo_only_when_requested(schedule: str, expected_rate: float) -> None:
    """A changed PPO learning rate cannot overwrite fixed AMP settings."""
    alg, obs = build_amp(amp_learning_rate=3e-4, amp_lr_schedule=schedule, amp_num_updates=1)
    alg.schedule = "fixed"
    alg.learning_rate = 9e-3
    for group in alg.optimizer.param_groups:
        group["lr"] = alg.learning_rate
    collect_rollout(alg, obs)
    alg.update()
    assert all(group["lr"] == pytest.approx(expected_rate) for group in alg.amp_optimizer.param_groups)


@pytest.mark.parametrize("amp_limit", [None, 1e-3])
def test_discriminator_clipping_is_independent_and_can_be_disabled(amp_limit: float | None) -> None:
    """Inspect gradients consumed by Adam, including the case where AMP clipping is off."""
    torch.manual_seed(11)
    alg, obs = build_amp(amp_max_grad_norm=amp_limit, amp_num_updates=1)
    alg.max_grad_norm = 1e-12  # PPO's limit must never become the discriminator's limit.
    norms = []

    def inspect_gradients(optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        gradients = [
            p.grad.flatten() for group in optimizer.param_groups for p in group["params"] if p.grad is not None
        ]
        norms.append(float(torch.cat(gradients).norm()))

    handle = alg.amp_optimizer.register_step_pre_hook(inspect_gradients)
    collect_rollout(alg, obs)
    alg.update()
    handle.remove()
    assert len(norms) == 1
    if amp_limit is None:
        assert norms[0] > 1e-3
    else:
        assert 0 < norms[0] <= amp_limit * 1.001


def test_amp_resume_restores_optimizer_rate_and_keeps_current_cadence_config() -> None:
    """Restore the independent optimizer while leaving explicit run configuration under caller control."""
    alg, obs = build_amp(amp_learning_rate=4e-4, amp_num_updates=3, amp_batch_size=5, amp_max_grad_norm=0.2)
    collect_rollout(alg, obs)
    alg.update()
    state = deepcopy(alg.save())
    assert state["amp_training_config"] == {
        "amp_learning_rate": pytest.approx(4e-4),
        "amp_lr_schedule": "fixed",
        "amp_num_updates": 3,
        "amp_batch_size": 5,
        "amp_max_grad_norm": 0.2,
    }
    restored, obs = build_amp(amp_num_updates=2, amp_batch_size=7)
    restored.load(state)
    assert restored.amp_learning_rate == pytest.approx(4e-4)
    assert restored.amp_num_updates == 2
    assert restored.amp_batch_size == 7
    assert restored.amp_max_grad_norm is None
    collect_rollout(restored, obs)
    restored.update()
    assert restored.amp_optimizer.param_groups[0]["lr"] == pytest.approx(4e-4)
    assert int(next(iter(restored.amp_optimizer.state.values()))["step"]) == 5
    # Existing migration pilot checkpoints do not contain the new audit metadata.
    state.pop("amp_training_config")
    restored.load(state)
    assert restored.amp_learning_rate == pytest.approx(4e-4)


def test_amp_actor_only_load_does_not_restore_discriminator_optimizer() -> None:
    """Playback must preserve its own unused AMP optimizer and discriminator settings."""
    alg, obs = build_amp(amp_learning_rate=4e-4)
    collect_rollout(alg, obs)
    alg.update()
    state = deepcopy(alg.save())
    inference, _ = build_amp(amp_learning_rate=8e-4)
    discriminator_before = deepcopy(inference.discriminator.state_dict())
    assert not inference.load(state, {"actor": True})
    assert inference.amp_optimizer.param_groups[0]["lr"] == pytest.approx(8e-4)
    assert not inference.amp_optimizer.state
    for key, value in inference.discriminator.state_dict().items():
        assert torch.equal(value, discriminator_before[key])


def test_amp_follow_mode_uses_restored_ppo_rate_without_overwriting_run_config() -> None:
    """Following PPO after resume uses its restored optimizer rate rather than checkpoint AMP metadata."""
    alg, obs = build_amp(amp_learning_rate=4e-4)
    collect_rollout(alg, obs)
    alg.update()
    state = deepcopy(alg.save())
    resumed, _ = build_amp(amp_lr_schedule="ppo")
    resumed.load(state)
    assert resumed.amp_lr_schedule == "ppo"
    assert resumed.amp_optimizer.param_groups[0]["lr"] == pytest.approx(resumed.learning_rate)
    assert resumed.amp_optimizer.param_groups[0]["lr"] == pytest.approx(
        state["optimizer_state_dict"]["param_groups"][0]["lr"]
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"amp_learning_rate": 0},
        {"amp_learning_rate": float("nan")},
        {"amp_lr_schedule": "adaptive"},
        {"amp_num_updates": 0},
        {"amp_num_updates": 1.5},
        {"amp_batch_size": 0},
        {"amp_max_grad_norm": 0},
        {"amp_max_grad_norm": float("inf")},
    ],
)
def test_invalid_amp_training_settings_fail_before_training(invalid: dict) -> None:
    """Reject mistyped or invalid training controls instead of silently using PPO defaults."""
    with pytest.raises(ValueError, match=next(iter(invalid))):
        build_amp(**invalid)


def test_independent_amp_settings_are_forwarded_with_algorithm_precedence() -> None:
    """Expose the new controls through runner configuration without overwriting explicit algorithm settings."""
    cfg = {
        "algorithm": {"class_name": "PPO", "amp_batch_size": 9},
        "amp_learning_rate": 3e-4,
        "amp_lr_schedule": "fixed",
        "amp_num_updates": 3,
        "amp_batch_size": 5,
        "amp_max_grad_norm": None,
    }
    before = deepcopy(cfg)
    transformed = prepare_amp_config(cfg)
    assert cfg == before
    assert transformed["algorithm"]["amp_learning_rate"] == pytest.approx(3e-4)
    assert transformed["algorithm"]["amp_lr_schedule"] == "fixed"
    assert transformed["algorithm"]["amp_num_updates"] == 3
    assert transformed["algorithm"]["amp_batch_size"] == 9
    assert transformed["algorithm"]["amp_max_grad_norm"] is None
    assert "amp_batch_size" not in transformed
