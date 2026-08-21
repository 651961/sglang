# SPDX-License-Identifier: Apache-2.0
"""SGLang native pipeline for SenseNova-U1.5."""

from __future__ import annotations

import os
from glob import glob

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sensenova_u1_5 import (
    SenseNovaU1_5PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sensenova_u1_5 import (
    SenseNovaU1_5SamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_5 import (
    SenseNovaU1_5GenerationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SenseNovaU1_5Pipeline(ComposedPipelineBase):
    pipeline_name = "SenseNovaU1_5Pipeline"
    pipeline_config_cls = SenseNovaU1_5PipelineConfig
    sampling_params_cls = SenseNovaU1_5SamplingParams

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "SenseNova-U1.5 currently supports monolithic SGLang Diffusion "
                "workers only; encoder/denoiser/decoder disaggregation is not "
                "implemented."
            )

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        if loaded_modules is not None and {"transformer", "tokenizer"}.issubset(
            loaded_modules
        ):
            return loaded_modules

        from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            resolve_transformer_quant_load_spec,
        )
        from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
        from sglang.multimodal_gen.runtime.models.encoders.sensenova_u1_5 import (
            SenseNovaU1_5NativeModel,
        )
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        device = get_local_torch_device()
        if not os.path.isdir(self.model_path):
            self.model_path = maybe_download_model(
                self.model_path,
                revision=server_args.revision,
            )

        config = SenseNovaU1_5NativeModel.config_from_path(self.model_path)
        model_cls, _ = ModelRegistry.resolve_model_cls(["SenseNovaU1_5NativeModel"])
        weight_files = sorted(glob(os.path.join(self.model_path, "*.safetensors")))
        if not weight_files:
            raise FileNotFoundError(
                f"No safetensors weights found under {self.model_path!r}"
            )

        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=config,
            server_args=server_args,
            safetensors_list=weight_files,
            component_model_path=self.model_path,
            model_cls=model_cls,
            cls_name=model_cls.__name__,
            component_name="transformer",
        )
        dtype = quant_spec.param_dtype or torch.bfloat16
        component_starts_on_cpu = server_args.should_start_component_on_cpu(
            "transformer"
        )
        # Online FP8 needs a device-side post-load quantization pass. Other
        # layerwise-offloaded loads can materialize directly on CPU and avoid
        # transiently placing the complete model on the accelerator.
        checkpoint_load_device = (
            device
            if quant_spec.needs_device_weight_postprocess or not component_starts_on_cpu
            else torch.device("cpu")
        )
        weight_load_plan = WeightLoadPlan.for_component(
            checkpoint_load_device=checkpoint_load_device,
            needs_device_weight_postprocess=quant_spec.needs_device_weight_postprocess,
            component_starts_on_cpu=component_starts_on_cpu,
        )
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={
                "config": config,
                "quant_config": quant_spec.runtime_quant_config,
            },
            weight_dir_list=weight_files,
            device=device,
            hsdp_replicate_dim=1,
            hsdp_shard_dim=1,
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            component_starts_on_cpu=component_starts_on_cpu,
            fsdp_inference=False,
            strict=True,
            weight_load_plan=weight_load_plan,
            checkpoint_key_filter=model_cls.should_materialize_checkpoint_weight,
        )
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(model)
        model.eval()
        tokenizer_path = getattr(server_args, "tokenizer_path", None) or self.model_path
        tokenizer = get_tokenizer(
            tokenizer_path,
            tokenizer_mode=getattr(server_args, "tokenizer_mode", "auto"),
            tokenizer_backend=getattr(server_args, "tokenizer_backend", "huggingface"),
            trust_remote_code=bool(getattr(server_args, "trust_remote_code", False)),
            tokenizer_revision=getattr(server_args, "tokenizer_revision", None),
        )
        return {"transformer": model, "tokenizer": tokenizer}

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            SenseNovaU1_5GenerationStage(
                model=self.get_module("transformer"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "sensenova_u1_5_generation",
        )


EntryClass = [SenseNovaU1_5Pipeline]
