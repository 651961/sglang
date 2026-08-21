# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for SenseNova-U1.5."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


@dataclass
class SenseNovaU1_5PipelineConfig(PipelineConfig):
    """Configuration for the native U1.5 pipeline."""

    task_type: ModelTaskType = ModelTaskType.TI2I
    dit_precision: str = "bf16"
    enable_autocast: bool = False

    def supports_dynamic_batching(self) -> bool:
        return False

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("memory",),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )

    def validate_server_args(self, server_args) -> None:
        if server_args.quantization not in (None, "fp8"):
            raise ValueError("SenseNova-U1.5 supports BF16 or online FP8 quantization.")
        num_gpus = int(server_args.num_gpus or 1)
        dp_size = int(server_args.dp_size or 1)
        tp_size = int(server_args.tp_size or 1)
        if tp_size != 1:
            raise ValueError(
                "SenseNova-U1.5 supports data parallel replicas, but not "
                "tensor parallelism; set --tp-size 1."
            )
        if server_args.use_fsdp_inference:
            raise ValueError(
                "SenseNova-U1.5 supports data parallel replicas, but not "
                "FSDP inference; disable --use-fsdp-inference."
            )
        if num_gpus != dp_size:
            raise ValueError(
                "SenseNova-U1.5 currently supports pure data parallelism: "
                "--num-gpus must equal --dp-size (one complete model per GPU)."
            )
        if int(server_args.sp_degree or 1) != 1:
            raise ValueError(
                "SenseNova-U1.5 data parallel deployment does not support "
                "sequence parallelism; set --sp-degree 1."
            )
        if int(server_args.cfg_parallel_degree or 1) != 1:
            raise ValueError(
                "SenseNova-U1.5 performs CFG inside each replica; disable "
                "CFG parallelism with --cfg-parallel-size 1."
            )
