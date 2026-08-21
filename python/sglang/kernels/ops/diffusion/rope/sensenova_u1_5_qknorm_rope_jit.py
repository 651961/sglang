from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "sensenova_u1_5_qknorm_rope_kv",
        cuda_files=["diffusion/sensenova_u1_5_qknorm_rope.cuh"],
        cuda_wrappers=[
            (
                "sensenova_u1_5_qknorm_rope_kv",
                "sensenova_u1_5::QKNormRopeKVKernel::run",
            )
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_sensenova_u1_5_qknorm_rope_kv(
    dtype: torch.dtype, device: torch.device
) -> bool:
    if dtype is not torch.bfloat16 or device.type != "cuda":
        return False
    try:
        _jit_module()
        return True
    except Exception as e:
        logger.warning("Failed to load SenseNova-U1.5 QKNorm+RoPE+KV kernel: %s", e)
        return False


@register_custom_op(mutates_args=["q", "cache_k", "cache_v"])
def fused_sensenova_u1_5_qknorm_rope_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    q_t_weight: torch.Tensor,
    k_t_weight: torch.Tensor,
    q_hw_weight: torch.Tensor,
    k_hw_weight: torch.Tensor,
    t_cos: torch.Tensor,
    t_sin: torch.Tensor,
    h_cos: torch.Tensor,
    h_sin: torch.Tensor,
    w_cos: torch.Tensor,
    w_sin: torch.Tensor,
    eps: float,
) -> None:
    """Fuse U1.5's two QK norms, three-axis RoPE, and generation KV pack."""
    _jit_module().sensenova_u1_5_qknorm_rope_kv(
        q,
        k,
        v,
        cache_k,
        cache_v,
        q_t_weight,
        k_t_weight,
        q_hw_weight,
        k_hw_weight,
        t_cos,
        t_sin,
        h_cos,
        h_sin,
        w_cos,
        w_sin,
        eps,
    )
