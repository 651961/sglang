# SPDX-License-Identifier: Apache-2.0
"""Fixed-step scheduler used by the Qwen-Image 2.1 PDD adapter."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class QwenImage21PDDScheduler(FlowMatchEulerDiscreteScheduler):
    """Flow-Match Euler with the already-shifted Qwen PDD sigma grid.

    Qwen's PDD exporter stores the resolution-dependent shifted schedule in
    ``pdd_config.json``.  The regular scheduler applies that shift again when
    explicit sigmas are supplied, so this class deliberately bypasses all
    schedule transformations and only appends the matching next sigma.
    """

    def __init__(self, pdd_sigmas, **kwargs):
        super().__init__(**kwargs)
        values = np.asarray(pdd_sigmas, dtype=np.float32)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Qwen-Image 2.1 PDD requires at least one sigma interval")
        if not np.isfinite(values).all() or np.any(values[:-1] < values[1:]):
            raise ValueError("Qwen-Image 2.1 PDD sigmas must be finite and descending")
        self.pdd_sigmas = values

    @classmethod
    def from_scheduler(cls, scheduler, pdd_sigmas):
        # Recreate the native scheduler from its loaded config, preserving all
        # runtime behavior of FlowMatchEulerDiscreteScheduler.step().
        params = inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters
        config = dict(scheduler.config)
        kwargs = {
            name: config[name] for name in params if name != "self" and name in config
        }
        return cls(pdd_sigmas=pdd_sigmas, **kwargs)

    @property
    def pdd_num_steps(self) -> int:
        return int(self.pdd_sigmas.size - 1)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas=None,
        mu: float | None = None,
        timesteps=None,
    ) -> None:
        if timesteps is not None:
            raise ValueError("Qwen-Image 2.1 PDD accepts a fixed sigma schedule only")

        if sigmas is None:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps or sigmas must be provided")
            count = int(num_inference_steps)
            selected = self.pdd_sigmas[:count]
        else:
            selected = np.asarray(sigmas, dtype=np.float32)
            count = int(selected.size)
            if selected.ndim != 1:
                raise ValueError("Qwen-Image 2.1 PDD sigmas must be one-dimensional")
            if num_inference_steps is not None and count != int(num_inference_steps):
                raise ValueError(
                    "sigmas must have the same length as num_inference_steps"
                )

        if not 1 <= count <= self.pdd_num_steps:
            raise ValueError(
                "Qwen-Image 2.1 PDD supports "
                f"1..{self.pdd_num_steps} steps, got {count}"
            )
        expected = self.pdd_sigmas[:count]
        if not np.allclose(selected, expected, rtol=1e-5, atol=1e-6):
            raise ValueError("The requested sigma schedule does not match Qwen PDD")

        # Keep the next sigma from the same exported grid.  For the normal
        # four-step request this is the terminal zero; warmup requests may use
        # a shorter prefix and therefore retain their real next sigma.
        full = np.concatenate([expected, self.pdd_sigmas[count : count + 1]])
        sigmas_tensor = torch.tensor(full, dtype=torch.float32, device=device)
        self.num_inference_steps = count
        self.timesteps = sigmas_tensor[:-1] * self.config.num_train_timesteps
        self.sigmas = sigmas_tensor
        self.sigma_max = float(full[0])
        self.sigma_min = float(full[-1])
        self._step_index = None
        self._begin_index = None
