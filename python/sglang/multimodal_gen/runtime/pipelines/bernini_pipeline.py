# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native SGLang inference pipelines for Bernini and Bernini-R."""

from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.bernini import (
    BerniniConfig,
    BerniniRConfig,
)
from sglang.multimodal_gen.configs.sample.bernini import (
    BerniniRSamplingParams,
    BerniniSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bernini import (
    BerniniDenoisingStage,
    BerniniPlanningStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bernini_r import (
    BerniniRDenoisingStage,
    BerniniTimestepPreparationStage,
    BerniniVisualConditioningStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BERNINI_MODEL_INDEX = {
    "_class_name": "BerniniRendererPipeline",
    "_diffusers_version": "0.35.0.dev0",
    "tokenizer": ["transformers", "T5Tokenizer"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "transformer_2": ["diffusers", "WanTransformer3DModel"],
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
}


class BerniniRendererPipeline(LoRAPipeline, ComposedPipelineBase):
    """Bernini-R renderer built on the native Wan2.2-T2V-A14B modules."""

    pipeline_name = "BerniniRendererPipeline"
    pipeline_config_cls = BerniniRConfig
    sampling_params_cls = BerniniRSamplingParams
    is_video_pipeline = True
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "transformer_2",
        "scheduler",
    ]

    def _load_config(self) -> dict[str, object]:
        assert self.server_args is not None
        self._validate_server_args(self.server_args)
        self.model_path = maybe_download_model(self.model_path)
        logger.info("Bernini-R model path: %s", self.model_path)
        # Bernini-R ships Diffusers components but no model_index.json.
        return dict(BERNINI_MODEL_INDEX)

    @staticmethod
    def _validate_server_args(server_args: ServerArgs) -> None:
        sp_degree = getattr(server_args, "sp_degree", 1)
        ulysses_degree = getattr(server_args, "ulysses_degree", 1)
        ring_degree = getattr(server_args, "ring_degree", 1)
        if sp_degree > 1 and (ulysses_degree != sp_degree or ring_degree != 1):
            raise ValueError(
                "Bernini sequence parallelism currently requires pure Ulysses: "
                "--ulysses-degree must equal --sp-degree and --ring-degree must be 1"
            )
        unsupported = {
            "--enable-cfg-parallel": server_args.enable_cfg_parallel,
            "disaggregated serving": getattr(server_args, "disagg_mode", False),
            "--enable-torch-compile": getattr(
                server_args, "enable_torch_compile", False
            ),
            "--enable-breakable-cuda-graph": getattr(
                server_args, "enable_breakable_cuda_graph", False
            ),
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(f"Bernini does not support: {', '.join(enabled)}")

        backend = str(getattr(server_args, "attention_backend", "") or "").lower()
        if any(name in backend for name in ("sparse", "sliding_tile", "vmoba")):
            raise ValueError("Bernini currently requires a dense attention backend")

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_stage(BerniniVisualConditioningStage(self.get_module("vae")))
        self.add_standard_latent_preparation_stage()
        self.add_stage(BerniniTimestepPreparationStage(self.get_module("scheduler")))

        def create_denoising_stage() -> BerniniRDenoisingStage:
            return BerniniRDenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            )

        self.add_stage_factory(
            RoleType.DENOISER,
            create_denoising_stage,
            "denoising_stage",
        )
        self.add_standard_decoding_stage()


class BerniniPipeline(LoRAPipeline, ComposedPipelineBase):
    """Bernini semantic planner and Wan2.2 renderer."""

    pipeline_name = "BerniniPipeline"
    pipeline_config_cls = BerniniConfig
    sampling_params_cls = BerniniSamplingParams
    is_video_pipeline = True
    _required_config_modules = [
        "processor",
        "text_encoder",
        "text_encoder_2",
        "tokenizer_2",
        "vae",
        "transformer",
        "transformer_2",
        "scheduler",
    ]

    def _load_config(self) -> dict[str, object]:
        assert self.server_args is not None
        BerniniRendererPipeline._validate_server_args(self.server_args)
        return super()._load_config()

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_stage(
            BerniniVisualConditioningStage(self.get_module("vae"), with_planner=True)
        )
        self.add_stage(
            BerniniPlanningStage(
                planner=self.get_module("text_encoder"),
                processor=self.get_module("processor"),
                t5_encoder=self.get_module("text_encoder_2"),
                t5_tokenizer=self.get_module("tokenizer_2"),
            )
        )
        self.add_standard_latent_preparation_stage()
        self.add_stage(BerniniTimestepPreparationStage(self.get_module("scheduler")))

        def create_denoising_stage() -> BerniniDenoisingStage:
            return BerniniDenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            )

        self.add_stage_factory(
            RoleType.DENOISER,
            create_denoising_stage,
            "denoising_stage",
        )
        self.add_standard_decoding_stage()


EntryClass = [BerniniRendererPipeline, BerniniPipeline]
