# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.schedulers.qwen_image21_pdd import (
    QwenImage21PDDScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21DenoisingStage,
    QwenImage21EncodingStage,
    QwenImage21InputValidationStage,
    prepare_qwen21_mu,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class QwenImage21Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "QwenImage21Pipeline"
    _required_config_modules = [
        "processor",
        "text_encoder",
        "transformer",
        "vae",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        pdd_path = envs.SGLANG_DIFFUSION_QWEN_IMAGE21_PDD_LORA
        if not pdd_path:
            return

        checkpoint = Path(pdd_path).expanduser().resolve()
        config_path = checkpoint.with_name("pdd_config.json")
        if not config_path.is_file():
            raise FileNotFoundError(
                "SGLANG_DIFFUSION_QWEN_IMAGE21_PDD_LORA requires pdd_config.json "
                f"next to the checkpoint: {config_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pdd_sigmas = config.get("pdd_sigmas")
        if not isinstance(pdd_sigmas, list):
            raise ValueError(f"pdd_config.json has no pdd_sigmas list: {config_path}")

        scheduler = self.get_module("scheduler")
        if scheduler is None:
            raise ValueError("Qwen-Image 2.1 PDD requires a scheduler module")
        self.modules["scheduler"] = QwenImage21PDDScheduler.from_scheduler(
            scheduler, pdd_sigmas
        )
        server_args.pipeline_config.pdd_sigmas = tuple(pdd_sigmas)

    def create_pipeline_stages(self, server_args):
        self.add_stage(QwenImage21InputValidationStage())
        self.add_stage_factory(
            RoleType.ENCODER,
            lambda: QwenImage21EncodingStage(
                self.get_module("text_encoder"),
                self.get_module("processor"),
                self.get_module("vae"),
                self.get_module("scheduler"),
            ),
            "conditioning_stage",
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_qwen21_mu]
        )
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: QwenImage21DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "denoising_stage",
        )
        self.add_standard_decoding_stage()


EntryClass = QwenImage21Pipeline
