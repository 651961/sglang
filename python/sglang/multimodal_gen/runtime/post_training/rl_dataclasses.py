# SPDX-License-Identifier: Apache-2.0
"""RL-specific dataclasses used by post-training and rollout paths."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RolloutSessionData:
    """Per-batch rollout state created by prepare_rollout(), lives on the batch object.

    Cleared by setting ``batch._rollout_session_data = None``.
    """

    pipeline_config: Any = None
    sigma_max: float = 0.0
    latents_shape: tuple | None = None
    noise_buffer: torch.Tensor | None = None

    local_log_prob_sum: list[torch.Tensor] = field(default_factory=list)
    local_log_prob_count: list[torch.Tensor] = field(default_factory=list)

    local_variance_noises: list[torch.Tensor] = field(default_factory=list)
    local_prev_sample_means: list[torch.Tensor] = field(default_factory=list)
    local_noise_std_devs: list[torch.Tensor] = field(default_factory=list)
    local_model_outputs: list[torch.Tensor] = field(default_factory=list)


@dataclass
class RolloutDebugTensors:
    """Container for rollout debug tensors collected during denoising."""

    rollout_variance_noises: torch.Tensor | None = None
    rollout_prev_sample_means: torch.Tensor | None = None
    rollout_noise_std_devs: torch.Tensor | None = None
    rollout_model_outputs: torch.Tensor | None = None
    step_indices: torch.Tensor | None = None


@dataclass
class RolloutDenoisingEnv:
    image_kwargs: dict[str, Any] | None = None
    pos_cond_kwargs: dict[str, Any] | None = None
    neg_cond_kwargs: dict[str, Any] | None = None
    guidance: torch.Tensor | None = None


@dataclass
class RolloutDitTrajectory:
    # [B, T+1, ...]: per-step noisy latents x_{t_0..t_{T-1}} followed by the
    # final denoised latent x_{t_T} (last scheduler.step output).
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None  # [T]
    # [T+1] scheduler.sigmas snapshot (post-shift, includes terminal 0).
    sigmas: torch.Tensor | None = None


@dataclass
class RolloutTransitionPairs:
    """Selected stochastic transitions used by policy-gradient training.

    Unlike ``RolloutDitTrajectory``, these tensors do not imply that adjacent
    entries are consecutive denoising states. Each row is an explicit
    ``(x_i, x_{i+1})`` pair identified by ``step_indices``. ``next_latents``
    may retain the sampler's fp32 action before the model-input dtype cast so
    trainer-side log-prob recomputation matches the rollout policy exactly.
    """

    step_indices: torch.Tensor | None = None  # [K]
    latents: torch.Tensor | None = None  # [B, K, ...]
    next_latents: torch.Tensor | None = None  # [B, K, ...]
    timesteps: torch.Tensor | None = None  # [K], model-native time coordinates
    next_timesteps: torch.Tensor | None = None  # [K], model-native coordinates
    sigmas: torch.Tensor | None = None  # [K], normalized flow sigma
    next_sigmas: torch.Tensor | None = None  # [K], normalized flow sigma
    base_noise_scale: float | None = None


@dataclass
class RolloutTrajectoryData:
    rollout_log_probs: torch.Tensor | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: RolloutDenoisingEnv | None = None
    dit_trajectory: RolloutDitTrajectory | None = None
    transition_pairs: RolloutTransitionPairs | None = None
