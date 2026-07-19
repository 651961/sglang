# SPDX-License-Identifier: Apache-2.0

"""Native SGLang pipeline configuration for Bernini and Bernini-R."""

from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders import T5Config
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_T2V_A14B_Config,
    WanI2VCommonConfig,
)


@dataclass
class BerniniRConfig(Wan2_2_T2V_A14B_Config, WanI2VCommonConfig):
    """Bernini-R reuses the two-expert Wan2.2-T2V-A14B architecture."""

    task_type: ModelTaskType = ModelTaskType.TI2V
    skip_input_image_preprocess: bool = True

    flow_shift: float | None = 5.0
    generator_device: str | None = "cpu"
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    max_source_id: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.vae_config.use_parallel_encode = False

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        del prompt_dtype
        # Official Bernini-R samples and advances the scheduler in FP32.
        return torch.float32

    def shard_latents_for_sp(self, batch, latents):
        # Bernini packs the complete target and source latents, then shards the
        # resulting token sequence in WanTransformer3DModel._forward_packed().
        del batch
        return latents, False

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
            auto_enable_cfg_parallel=False,
        )


@dataclass
class BerniniConfig(BerniniRConfig):
    """Bernini adds a Qwen2.5-VL planner before the renderer."""

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen2_5VLConfig(), T5Config())
    )
    text_encoder_precisions: tuple[str, ...] = ("bf16", "bf16")
    preprocess_text_funcs: tuple = (None, None)
    postprocess_text_funcs: tuple = (None, None)
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{}, {}])
