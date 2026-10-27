# AMP extension

This fork adds AMP on top of the current upstream PPO, actor/critic models, and TensorDict observation API.
Upstream PPO and OnPolicyRunner implementations remain the source of the base learning loop.
The extension is adapted from the AMP implementation in [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab).

## Environment contract

The environment supplies its usual actor/critic observation groups and one uncorrupted feature group,
`obs["amp"]`, with shape `[num_envs, feature_dim]`. AMPPPO remembers the pre-action feature and pairs it with
the post-action feature. The default name can be changed with `amp_observation_group`.

On automatic reset, the returned observation may already belong to a new episode. Done transitions are
excluded from discriminator replay and retain their full task reward unless the environment supplies
`extras["amp_terminal_observations"]` with shape `[num_envs, feature_dim]`. An optional boolean
`extras["amp_terminal_mask"]` identifies done environments for which real terminal features are available.
These must be observations from before reset, with the same schema as `obs["amp"]`.

## Configuration and API

Use `rsl_rl.algorithms:AMPPPO` as the algorithm class in a current OnPolicyRunner configuration.
Its constructor takes the current actor, critic, and rollout storage objects plus these keyword arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `amp_observation_group` | `"amp"` | Single-frame feature observation group |
| `amp_reward_coef` | `0.1` | Motion-style reward scale |
| `amp_task_reward_lerp` | `0.75` | Task fraction in the final reward blend |
| `amp_discr_hidden_dims` | `(1024, 512, 256)` | ReLU discriminator hidden widths |
| `amp_replay_buffer_size` | `100000` | Maximum number of policy feature pairs |
| `amp_grad_penalty` | `10.0` | Squared expert input-gradient penalty weight |
| `amp_learning_rate` | `0.001` | Independent discriminator Adam learning rate |
| `amp_lr_schedule` | `"fixed"` | Keep the AMP rate fixed, or explicitly select `"ppo"` to follow PPO's final rate |
| `amp_num_updates` | `20` | Discriminator optimizer steps per rollout, independent of PPO epochs/mini-batches |
| `amp_batch_size` | `512` | Policy transition pairs and expert pairs sampled per discriminator step, for each source |
| `amp_max_grad_norm` | `None` | Optional AMP-only gradient-norm clipping limit; `None` disables clipping |
| `amp_motion_files` | `""` | NPZ file or directory searched recursively |
| `amp_body_names` | `()` | Ordered selected bodies for raw archives |
| `amp_anchor_name` | `""` | Anchor body for raw archives |
| `amp_all_body_names` | `()` | Archive body order when metadata is absent |
| `amp_frame_dt` | `None` | Control interval; validates clip frame sampling |
| `amp_data` | `None` | Optional sampler with `observation_dim` and `sample(batch_size)` |
| `min_std` | `None` | Optional lower bound for state-independent Gaussian action std |

`AmpOnPolicyRunner` and `prepare_amp_config` also accept original top-level AMP settings and move them into
the algorithm configuration without mutating the caller. `min_normalized_std` maps to `min_std`.
`amp_num_preload_transitions` is obsolete: complete feature clips are cached once instead of preloading
duplicated random transition pairs.

```python
from rsl_rl.runners import AmpOnPolicyRunner

# cfg contains the current upstream actor, critic, algorithm, and runner settings.
cfg["obs_groups"] = {"actor": ["actor"], "critic": ["critic"]}
cfg["amp_motion_files"] = "/path/to/precomputed_motion_clips"
cfg["amp_frame_dt"] = 0.02
cfg["amp_reward_coef"] = 0.1
cfg["amp_task_reward_lerp"] = 0.75
cfg["amp_learning_rate"] = 0.001
cfg["amp_num_updates"] = 20
cfg["amp_batch_size"] = 512
cfg["amp_max_grad_norm"] = None
runner = AmpOnPolicyRunner(env, cfg, log_dir="logs/amp", device="cuda:0")
runner.learn(num_learning_iterations=100)
```

Simulator-specific adapters belong in the consuming project. This library imports neither mjlab nor its robot assets.

## Expert data

For simulator-independent precomputed data, NPZ archives contain:

- `fps`: positive scalar frame rate.
- `amp_observations`: finite `[frames, feature_dim]` float features with the exact live observation schema.

Raw AMP_mjlab archives use `body_pos_w`, `body_quat_w` (wxyz), `body_lin_vel_w`, and `body_ang_vel_w`.
Their body order comes from string `body_names` metadata or the explicit `amp_all_body_names` configuration.
Names without metadata must be verified against the current robot model by the consuming project.

`compute_amp_features` is shared by the raw loader and live environment adapters. Its order is all
anchor-relative positions, all relative 6D rotations (first two matrix columns, row-major flatten), all
body-local linear velocities, then all body-local angular velocities. Each body contributes 15 features.

Sampling chooses clips uniformly, then samples a valid within-clip pair. `fps * amp_frame_dt` must be a
positive integer stride; incompatible intervals raise an error requiring resampling. Clip boundaries
are never paired. Raw and precomputed archives must agree in feature dimensions, body order, coordinates,
and temporal interval.

## Learning and checkpoints

The least-squares classifier targets expert `+1` and policy `-1`, with expert gradient-norm regularization.
Rewards retain the original AMP blend:

```text
style = amp_reward_coef * max(0, 1 - 0.25 * (D - 1)^2)
combined = (1 - amp_task_reward_lerp) * style + amp_task_reward_lerp * task_reward
```

AMPPPO delegates PPO updates to upstream, then trains the discriminator using a separate Adam optimizer.
By default its learning rate, update count, batch size, and clipping are independent of PPO. These are
starting configuration values, not empirically tuned gait-quality settings. `amp_lr_schedule="ppo"` explicitly
selects the behavior of the first migration pilot: use PPO's final learning rate for the subsequent AMP phase.
It does not reproduce the legacy per-mini-batch learning-rate sequence. AMP clipping is disabled by default;
`amp_max_grad_norm` enables it with an AMP-only threshold. Non-finite discriminator gradients still raise an
error when clipping is disabled.

This avoids copying PPO internals; optimization details are not intended to reproduce the legacy
implementation bit for bit. Specifically, the legacy implementation interleaves PPO and discriminator
updates in each mini-batch using one Adam with disjoint parameter groups. The legacy code evaluates its gradient penalty on raw
expert features; the current implementation uses normalized expert features. Legacy running moments are
updated from already-normalized current policy/expert frames; current moments use raw current and next
frames from both sources. These are training-behavior changes, not requirements of the new upstream API,
and integration tests do not establish training equivalence or improved gait quality.

Running AMP moments are torch buffers and are updated explicitly after discriminator steps.
Distributed parameter and moment synchronization is included;
the consuming project's current real-simulation validation covers a single GPU.

Full checkpoints add `discriminator_state_dict`, `amp_normalizer_state_dict`, `amp_optimizer_state_dict`,
`amp_observation_dim`, and audit metadata `amp_training_config` to upstream state. Full resume requires matching AMP state and configuration;
actor-only loading remains available for inference. Replay and expert caches are rebuilt, so resume does
not restore the exact random trajectory. A partial `load_cfg` restores AMP only when `"amp": True` is set;
`load_cfg=None` restores the complete training state.

Like upstream PPO, restoring the optimizer also restores its moments and saved learning rate; for fixed
AMP scheduling this saved rate becomes `amp_learning_rate`. Following mode uses the restored PPO rate.
The active schedule, update count, batch size, and clipping are selected by the current run configuration;
checkpoint `amp_training_config` records the previous values for comparison and does not overwrite them.
To continue the same experiment, supply the same run settings. Older migration pilot checkpoints without
the audit metadata remain loadable. Actor-only playback does not restore AMP optimizer or discriminator state.

AMP reward diagnostics are written under `AMP/`; discriminator losses are returned alongside upstream losses.
Upstream episode reward logging still receives the original environment rewards; use `AMP/combined_reward`
to inspect the mixed per-step reward used by PPO.

## Validation

From a Python environment with this fork installed:

```bash
pytest tests/algorithms/test_amp.py -q
pytest tests -q
```

Tests cover the reward formula, real actor and discriminator updates, checkpoint state restoration,
actor-only compatibility, auto-reset boundaries, ring-buffer wraparound, clip-local time sampling,
feature ordering and rotated coordinate frames, legacy configuration translation, independent AMP update
counts/batch sizes/rates, optional gradient clipping, and optimizer-rate restoration without overwriting
the caller's cadence configuration.
