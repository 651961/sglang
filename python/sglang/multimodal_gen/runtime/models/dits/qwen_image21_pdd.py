"""Qwen-Image 2.1 PDD adapter loading for the native SGLang DiT.

The released Qwen-Image-2.1-Fun-Acc checkpoint is an inference-only PDD
bundle.  Its ordinary transformer weights are represented by low-rank deltas,
some RMSNorm weights are stored as full parameters, and ``proj_out.weight`` is
already exported as one fused head per denoising interval.  This is different
from a regular PEFT adapter and must be installed after the base transformer
weights have been materialized.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _resolve_child(root: nn.Module, name: str) -> nn.Module:
    try:
        return root.get_submodule(name)
    except AttributeError as exc:
        raise ValueError(f"Qwen-Image 2.1 PDD target does not exist: {name}") from exc


class QwenImage21PDDHead(nn.Module):
    """Select one prefused output projection by the current denoise index."""

    def __init__(self, source: nn.Linear, weights: torch.Tensor) -> None:
        super().__init__()
        if weights.ndim != 3:
            raise ValueError(
                "Qwen-Image 2.1 PDD proj_out.weight must have shape "
                "[steps, out_features, in_features]"
            )
        expected = (weights.shape[1], weights.shape[2])
        if tuple(source.weight.shape) != expected:
            raise ValueError(
                "Qwen-Image 2.1 PDD proj_out shape mismatch: "
                f"base={tuple(source.weight.shape)}, heads={tuple(weights.shape)}"
            )
        if source.bias is not None:
            raise ValueError("Qwen-Image 2.1 PDD expects a bias-free proj_out")
        self.weight = nn.Parameter(
            weights.to(device=source.weight.device, dtype=source.weight.dtype),
            requires_grad=False,
        )

    @property
    def num_steps(self) -> int:
        return int(self.weight.shape[0])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        step = int(get_forward_context().current_timestep)
        if not 0 <= step < self.num_steps:
            raise ValueError(
                "Qwen-Image 2.1 PDD has "
                f"{self.num_steps} fused heads but the denoising loop is at "
                f"step {step}; use the matching PDD step count."
            )
        weight = self.weight[step].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        # Converted Qwen PDD bundles were exported with native-time FP32
        # prediction state; match the reference helper's transformer forward
        # hook before the scheduler computes the Euler update.
        return F.linear(hidden_states, weight).float()


def load_qwen_image21_pdd(transformer: nn.Module, path: str) -> None:
    """Merge a released Qwen-Image 2.1 PDD bundle into ``transformer``.

    The current implementation intentionally supports the single-process
    native Qwen path.  The PDD bundle contains full (unsharded) LoRA matrices;
    adding tensor-parallel slicing requires a separate conversion step.
    """

    pdd_path = Path(path).expanduser().resolve()
    if not pdd_path.is_file():
        raise FileNotFoundError(f"Qwen-Image 2.1 PDD checkpoint not found: {pdd_path}")
    config_path = pdd_path.with_name("pdd_config.json")
    if not config_path.is_file():
        raise FileNotFoundError(f"Qwen-Image 2.1 PDD config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("pdd_export_format") != "qwenimage21_extracted_prefused_v1":
        raise ValueError(
            "Unsupported Qwen-Image 2.1 PDD export format: "
            f"{config.get('pdd_export_format')!r}"
        )
    if (
        not config.get("pdd_inference_only")
        or int(config.get("pdd_block_size", 0)) != 1
        or config.get("pdd_sampling_precision") != "native_time_fp32_state"
    ):
        raise ValueError(
            "SGLang Qwen-Image 2.1 PDD requires the inference-only, "
            "single-interval exported bundle"
        )
    state = load_file(str(pdd_path), device="cpu")
    lora_targets = [
        target.strip()
        for target in str(config.get("lora_targets", "")).split(",")
        if target.strip()
    ]
    rank = int(config["lora_rank"])
    alpha = float(config["lora_alpha"])
    scale = alpha / rank

    if getattr(transformer, "_qwen21_pdd_loaded", False):
        raise RuntimeError("Qwen-Image 2.1 PDD is already loaded on this transformer")

    expected_lora = {
        f"{target}.{suffix}"
        for target in lora_targets
        for suffix in ("lora_down", "lora_up")
    }
    full_parameters = set(config.get("pdd_full_parameters", []))
    expected = expected_lora | full_parameters | {"proj_out.weight"}
    actual = set(state)
    if actual != expected:
        raise ValueError(
            "Qwen-Image 2.1 PDD checkpoint keys differ: "
            f"missing={sorted(expected - actual)[:8]}, "
            f"extra={sorted(actual - expected)[:8]}"
        )
    head_steps = int(state["proj_out.weight"].shape[0])
    config_steps = int(config.get("pdd_num_steps", head_steps))
    if (
        head_steps != config_steps
        or len(config.get("pdd_sigmas", [])) != head_steps + 1
    ):
        raise ValueError(
            "Qwen-Image 2.1 PDD head count and sigma grid are inconsistent: "
            f"heads={head_steps}, config_steps={config_steps}, "
            f"sigmas={len(config.get('pdd_sigmas', []))}"
        )

    # Merge ordinary LoRA targets into the already-loaded base weights.
    for target in lora_targets:
        module = _resolve_child(transformer, target)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(f"Qwen-Image 2.1 PDD target has no weight: {target}")
        down = state[f"{target}.lora_down"]
        up = state[f"{target}.lora_up"]
        delta = (up.float() @ down.float()) * scale
        if tuple(delta.shape) != tuple(weight.shape):
            raise ValueError(
                "Qwen-Image 2.1 PDD currently requires a single unsharded process; "
                f"target {target} has base={tuple(weight.shape)} and "
                f"delta={tuple(delta.shape)}"
            )
        with torch.no_grad():
            weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))

    # Restore the full parameters exported by the PDD conversion.
    for target in full_parameters:
        parameter = transformer.get_parameter(target)
        value = state[target]
        if tuple(parameter.shape) != tuple(value.shape):
            raise ValueError(
                f"Qwen-Image 2.1 PDD full parameter mismatch: {target}, "
                f"base={tuple(parameter.shape)}, bundle={tuple(value.shape)}"
            )
        with torch.no_grad():
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    source = getattr(transformer, "proj_out", None)
    if not isinstance(source, nn.Linear):
        raise ValueError("Qwen-Image 2.1 PDD expects an ordinary nn.Linear proj_out")
    transformer.proj_out = QwenImage21PDDHead(source, state["proj_out.weight"])
    transformer._qwen21_pdd_loaded = True
    logger.info(
        "Qwen-Image 2.1 PDD loaded: %d steps, rank=%d, checkpoint=%s",
        transformer.proj_out.num_steps,
        rank,
        pdd_path,
    )
