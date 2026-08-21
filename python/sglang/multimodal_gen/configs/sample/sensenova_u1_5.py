# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SenseNova-U1.5 image generation and editing."""

from dataclasses import dataclass, field
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SenseNovaU1_5SamplingParams(SamplingParams):
    """Sampling parameters for the native SenseNova-U1.5 sampler."""

    data_type: DataType = field(default=DataType.IMAGE, init=False)
    _default_width: ClassVar[int] = 2048
    _default_height: ClassVar[int] = 2048

    cfg_scale: float = 4.0
    img_cfg_scale: float = 1.0
    cfg_norm: str = "none"
    timestep_shift: float = 3.0
    num_inference_steps: int = 50
    t_eps: float = 0.02
    target_pixels: int = 2048 * 2048
    input_max_pixels: int | str | None = "auto"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cfg_norm not in {"none", "global", "channel"}:
            raise ValueError("cfg_norm must be one of 'none', 'global', or 'channel'")
        if self.cfg_scale < 0 or self.img_cfg_scale < 0:
            raise ValueError("cfg_scale and img_cfg_scale must be non-negative")
        if self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be positive")
        if self.t_eps <= 0:
            raise ValueError("t_eps must be positive")
