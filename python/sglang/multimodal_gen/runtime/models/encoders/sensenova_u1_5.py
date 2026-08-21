# SPDX-License-Identifier: Apache-2.0
"""Native SenseNova-U1.5 Qwen3/MoT image generator."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.diffusion import (
    can_use_fused_rope_rotate_half,
    can_use_fused_sensenova_u1_5_qknorm_rope_kv,
    fused_sensenova_u1_5_qknorm_rope_kv,
    fused_rope_rotate_half_bitexact,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.multimodal.processors.qwen_vl import smart_resize

SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, "
    "you MUST start with a <think></think> block. Put all reasoning inside the block using plain "
    "text. DO NOT include any image tags. Keep it reasonable and directly useful for producing "
    "the final image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final "
    "image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a high-quality image "
    "based on the user's description.\n- Ensure visual clarity, semantic consistency, and "
    "completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n- Use the provided image(s) as input or reference for modification or "
    "transformation.\n- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\nGeneral Rules:\n- For any "
    "visible text in the image, follow the language specified for the rendered text in the user's "
    "description, not the language of the prompt. If no language is specified, use the user's "
    "input language."
)
DEFAULT_SYSTEM_MESSAGE = (
    "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，"
    "英文名叫InternVL, 是一个有用无害的人工智能助手。"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_NORMALIZE = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
IMAGE_TO_TENSOR = T.ToTensor()


def _namespace(values: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**values)


def _make_sensenova_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> nn.Module:
    if quant_config is None:
        return nn.Linear(in_features, out_features, bias=bias)
    return ReplicatedLinear(
        input_size=in_features,
        output_size=out_features,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
    )


def _linear_output(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    output = layer(hidden_states)
    return output[0] if isinstance(output, tuple) else output


def _make_sensenova_rms_norm(hidden_size: int, eps: float) -> RMSNorm:
    return RMSNorm(
        hidden_size,
        eps=eps,
        cast_x_before_out_mul=True,
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_axis_rope(
    positions: torch.Tensor, dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the broadcastable cos/sin tables for one RoPE axis."""
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
            / float(dim)
        )
    )
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def _apply_axis_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope
    if cos.dtype != q.dtype:
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    if q.dtype is torch.bfloat16 and q.is_cuda:
        # Spatial H/W slices are strided views. Materializing them lets all
        # three axes use the fused kernel instead of silently falling back.
        fused_q = q.contiguous()
        fused_k = k.contiguous()
        batch = q.shape[0]
        cos_rows = cos.expand(batch, -1, -1).reshape(-1, cos.shape[-1])
        sin_rows = sin.expand(batch, -1, -1).reshape(-1, sin.shape[-1])
        can_fuse = can_use_fused_rope_rotate_half(fused_q, cos_rows, sin_rows)
        can_fuse = can_fuse and can_use_fused_rope_rotate_half(
            fused_k, cos_rows, sin_rows
        )
        if can_fuse:
            return (
                fused_rope_rotate_half_bitexact(fused_q, cos_rows, sin_rows),
                fused_rope_rotate_half_bitexact(fused_k, cos_rows, sin_rows),
            )
    cos = cos[None, :, None]
    sin = sin[None, :, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@dataclass
class SenseNovaU1_5KVCache:
    """Preallocated FlashAttention KV storage for one decoder layer."""

    key: torch.Tensor
    value: torch.Tensor
    prefix_length: int


def _prefix_attention_runs(block_index: torch.Tensor) -> list[tuple[int, int, bool]]:
    """Split ``same-block | causal`` attention into mask-free FA calls.

    Runs of singleton (text) blocks are causal. Multi-token image blocks are
    bidirectional, while their K/V include all preceding runs.
    """
    values = block_index.tolist()
    runs: list[tuple[int, int, bool]] = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        if end - start > 1:
            runs.append((start, end, False))
            start = end
            continue
        end = start + 1
        while end < len(values):
            next_end = end + 1
            while next_end < len(values) and values[next_end] == values[end]:
                next_end += 1
            if next_end - end > 1:
                break
            end = next_end
        runs.append((start, end, True))
        start = end
    return runs


class SenseNovaU1_5Attention(nn.Module):
    """Qwen3 attention with separate understanding and image-generation weights."""

    def __init__(
        self,
        config: SimpleNamespace,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = int(config.head_dim)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        q_out = config.num_attention_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=bool(config.attention_bias),
            quant_config=quant_config,
            prefix=(f"language_model.model.layers.{layer_idx}.self_attn.qkv_proj"),
        )
        self.qkv_proj_mot_gen = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=bool(config.attention_bias),
            quant_config=quant_config,
            prefix=(
                f"language_model.model.layers.{layer_idx}.self_attn.qkv_proj_mot_gen"
            ),
        )

        def row(suffix: str) -> nn.Module:
            return _make_sensenova_linear(
                q_out,
                config.hidden_size,
                bias=bool(config.attention_bias),
                quant_config=quant_config,
                prefix=f"language_model.model.layers.{layer_idx}.self_attn.{suffix}",
            )

        self.o_proj = row("o_proj")
        self.o_proj_mot_gen = row("o_proj_mot_gen")

        half = self.head_dim // 2
        quarter = self.head_dim // 4
        self.q_norm = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        self.k_norm = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        self.q_norm_hw = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        self.k_norm_hw = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        self.q_norm_mot_gen = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        self.k_norm_mot_gen = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        self.q_norm_hw_mot_gen = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        self.k_norm_hw_mot_gen = _make_sensenova_rms_norm(half, config.rms_norm_eps)
        if half != quarter * 2:
            raise ValueError(
                f"SenseNova-U1.5 head_dim must be divisible by four: {self.head_dim}"
            )

        supported = (AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA)
        self.prefix_attention = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
            supported_attention_backends=supported,
        )
        self.prefix_block_attention = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=False,
            supported_attention_backends=supported,
        )
        self.generation_attention = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=False,
            supported_attention_backends=supported,
        )

    def _qkv(
        self,
        hidden_states: torch.Tensor,
        generation: bool,
        rope_cache: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self._project_qkv(hidden_states, generation)
        q, k = self._apply_qk_rope(q, k, generation, rope_cache)
        return q, k, v

    def _apply_qk_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        generation: bool,
        rope_cache: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_norm, k_norm, q_norm_hw, k_norm_hw = self._norms(generation)
        q_t, q_hw = q.chunk(2, dim=-1)
        k_t, k_hw = k.chunk(2, dim=-1)
        q_t, k_t = self._norm_rope(q_t, k_t, q_norm, k_norm, rope_cache[0])
        q_hw, k_hw = self._norm_rope(
            q_hw, k_hw, q_norm_hw, k_norm_hw, rope_cache[1], rope_dim=32
        )
        q_h, q_w = q_hw.split(self.head_dim // 4, dim=-1)
        k_h, k_w = k_hw.split(self.head_dim // 4, dim=-1)
        q_w, k_w = _apply_axis_rope(q_w, k_w, rope_cache[2])
        q[..., : self.head_dim // 2] = q_t
        k[..., : self.head_dim // 2] = k_t
        q[..., self.head_dim // 2 :] = torch.cat((q_h, q_w), dim=-1)
        k[..., self.head_dim // 2 :] = torch.cat((k_h, k_w), dim=-1)
        return q, k

    def _fused_generation_qk_rope_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_slot: torch.Tensor,
        value_slot: torch.Tensor,
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> bool:
        q_norm, k_norm, q_norm_hw, k_norm_hw = self._norms(True)
        weights = (q_norm.weight, k_norm.weight, q_norm_hw.weight, k_norm_hw.weight)
        caches = tuple(tensor for axis in rope_cache for tensor in axis)
        expected_shapes = ((q.shape[1], 64),) * 2 + ((q.shape[1], 32),) * 4
        if not (
            self.head_dim == 128
            and q.dtype is torch.bfloat16
            and all(
                tensor.is_cuda and tensor.dtype is q.dtype
                for tensor in (q, k, v, key_slot, value_slot, *weights, *caches)
            )
            and all(tensor.is_contiguous() for tensor in caches)
            and all(
                tensor.shape == shape for tensor, shape in zip(caches, expected_shapes)
            )
            and q_norm.variance_epsilon == q_norm_hw.variance_epsilon
            and k_norm.variance_epsilon == q_norm.variance_epsilon
            and k_norm_hw.variance_epsilon == q_norm.variance_epsilon
            and can_use_fused_sensenova_u1_5_qknorm_rope_kv(q.dtype, q.device)
        ):
            return False
        fused_sensenova_u1_5_qknorm_rope_kv(
            q,
            k,
            v,
            key_slot,
            value_slot,
            *weights,
            *caches,
            q_norm.variance_epsilon,
        )
        return True

    def _norms(self, generation: bool):
        if generation:
            return (
                self.q_norm_mot_gen,
                self.k_norm_mot_gen,
                self.q_norm_hw_mot_gen,
                self.k_norm_hw_mot_gen,
            )
        return self.q_norm, self.k_norm, self.q_norm_hw, self.k_norm_hw

    @staticmethod
    def _norm_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        rope_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = q.shape[-1]
        rope_dim = head_dim if rope_dim is None else rope_dim
        q, k = q_norm(q), k_norm(k)
        if rope_dim < head_dim:
            q_head, q_tail = q.split(rope_dim, dim=-1)
            k_head, k_tail = k.split(rope_dim, dim=-1)
            q_head, k_head = _apply_axis_rope(q_head, k_head, rope_cache)
            return torch.cat((q_head, q_tail), -1), torch.cat((k_head, k_tail), -1)
        return _apply_axis_rope(q, k, rope_cache)

    def _project_qkv(
        self, hidden_states: torch.Tensor, generation: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if generation:
            q_proj = self.qkv_proj_mot_gen
        else:
            q_proj = self.qkv_proj

        batch, seq_len, _ = hidden_states.shape
        qkv = _linear_output(q_proj, hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        return q, k, v

    def _output_projection(
        self, output: torch.Tensor, generation: bool
    ) -> torch.Tensor:
        output = output.reshape(*output.shape[:2], -1).contiguous()
        projection = self.o_proj_mot_gen if generation else self.o_proj
        return _linear_output(projection, output)

    def forward_prefix(
        self,
        hidden_states: torch.Tensor,
        attention_runs: list[tuple[int, int, bool]],
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self._qkv(hidden_states, generation=False, rope_cache=rope_cache)
        outputs = []
        for start, end, causal in attention_runs:
            attention = self.prefix_attention if causal else self.prefix_block_attention
            outputs.append(attention(q[:, start:end], k[:, :end], v[:, :end]))
        output = torch.cat(outputs, dim=1)
        return self._output_projection(output, generation=False), (k, v)

    def forward_generation(
        self,
        hidden_states: torch.Tensor,
        prefix_cache: SenseNovaU1_5KVCache,
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(hidden_states, generation=True)
        end = prefix_cache.prefix_length + k.shape[1]
        key_slot = prefix_cache.key[:, prefix_cache.prefix_length : end]
        value_slot = prefix_cache.value[:, prefix_cache.prefix_length : end]
        fused = self._fused_generation_qk_rope_kv(
            q, k, v, key_slot, value_slot, rope_cache
        )
        if not fused:
            q, k = self._apply_qk_rope(q, k, True, rope_cache)
            key_slot.copy_(k)
            value_slot.copy_(v)
        output = self.generation_attention(
            q, prefix_cache.key[:, :end], prefix_cache.value[:, :end]
        )
        return self._output_projection(output, generation=True)

    def forward_generation_branches(
        self,
        hidden_states: torch.Tensor,
        prefix_caches: list[SenseNovaU1_5KVCache],
        rope_caches: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    ) -> torch.Tensor:
        """Run all CFG branches through shared projection GEMMs."""
        branch_count = len(prefix_caches)
        q, k, v = self._project_qkv(hidden_states, generation=True)
        outputs = []
        for q_branch, k_branch, v_branch, cache, rope_cache in zip(
            q.chunk(branch_count),
            k.chunk(branch_count),
            v.chunk(branch_count),
            prefix_caches,
            rope_caches,
        ):
            end = cache.prefix_length + k_branch.shape[1]
            key_slot = cache.key[:, cache.prefix_length : end]
            value_slot = cache.value[:, cache.prefix_length : end]
            fused = self._fused_generation_qk_rope_kv(
                q_branch,
                k_branch,
                v_branch,
                key_slot,
                value_slot,
                rope_cache,
            )
            if not fused:
                q_branch, k_branch = self._apply_qk_rope(
                    q_branch, k_branch, True, rope_cache
                )
                key_slot.copy_(k_branch)
                value_slot.copy_(v_branch)
            outputs.append(
                self.generation_attention(
                    q_branch, cache.key[:, :end], cache.value[:, :end]
                )
            )
        return self._output_projection(torch.cat(outputs), generation=True)


class SenseNovaU1_5MLP(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = _make_sensenova_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = _linear_output(self.gate_up_proj, hidden_states)
        if (
            gate_up.is_cuda
            and gate_up.dtype in (torch.float16, torch.bfloat16)
            and gate_up.is_contiguous()
            and gate_up.shape[-1] % 32 == 0
            and gate_up.numel() > 0
        ):
            # Preserve the eager low-precision activation rounding.
            hidden_states = silu_and_mul_with_activation_rounding(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = torch.nn.functional.silu(gate) * up
        return _linear_output(self.down_proj, hidden_states)


class SenseNovaU1_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = SenseNovaU1_5Attention(config, layer_idx, quant_config)
        prefix = f"language_model.model.layers.{layer_idx}"
        self.mlp = SenseNovaU1_5MLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.mlp_mot_gen = SenseNovaU1_5MLP(
            config, quant_config, prefix=f"{prefix}.mlp_mot_gen"
        )
        self.input_layernorm = _make_sensenova_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = _make_sensenova_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.input_layernorm_mot_gen = _make_sensenova_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm_mot_gen = _make_sensenova_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        generation: bool,
        attention_runs: list[tuple[int, int, bool]] | None = None,
        prefix_cache: SenseNovaU1_5KVCache | None = None,
        residual: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor]
    ):
        if generation:
            if prefix_cache is None:
                raise ValueError("generation requires prefix_cache")
            return self.forward_generation(
                hidden_states, prefix_cache, rope_cache, residual
            )
        if attention_runs is None:
            raise ValueError("prefix requires attention runs")
        return self.forward_prefix(hidden_states, attention_runs, rope_cache, residual)

    def forward_prefix(
        self,
        hidden_states: torch.Tensor,
        attention_runs: list[tuple[int, int, bool]],
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        # Carry the residual across layers so RMSNorm can fuse the add with
        # normalization.  The representation is equivalent to the explicit
        # ``residual + attention`` / ``residual + mlp`` formulation.
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm(hidden_states)
        else:
            normalized, residual = self.input_layernorm(hidden_states, residual)
        attention, cache = self.self_attn.forward_prefix(
            normalized, attention_runs, rope_cache
        )
        normalized, residual = self.post_attention_layernorm(attention, residual)
        return self.mlp(normalized), residual, cache

    def forward_generation(
        self,
        hidden_states: torch.Tensor,
        prefix_cache: SenseNovaU1_5KVCache,
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm_mot_gen(hidden_states)
        else:
            normalized, residual = self.input_layernorm_mot_gen(hidden_states, residual)
        attention = self.self_attn.forward_generation(
            normalized, prefix_cache, rope_cache
        )
        normalized, residual = self.post_attention_layernorm_mot_gen(
            attention, residual
        )
        return self.mlp_mot_gen(normalized), residual

    def forward_generation_branches(
        self,
        hidden_states: torch.Tensor,
        prefix_caches: list[SenseNovaU1_5KVCache],
        rope_caches: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm_mot_gen(hidden_states)
        else:
            normalized, residual = self.input_layernorm_mot_gen(hidden_states, residual)
        attention = self.self_attn.forward_generation_branches(
            normalized,
            prefix_caches,
            rope_caches,
        )
        normalized, residual = self.post_attention_layernorm_mot_gen(
            attention, residual
        )
        return self.mlp_mot_gen(normalized), residual


class SenseNovaU1_5Qwen3Model(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                SenseNovaU1_5DecoderLayer(config, i, quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _make_sensenova_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.norm_mot_gen = _make_sensenova_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )

    def build_rope_cache(
        self, indexes: torch.Tensor, dtype: torch.dtype
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Build all three axis tables once for a prefix or denoise request."""
        return (
            _build_axis_rope(
                indexes[0], self.config.head_dim // 2, self.config.rope_theta, dtype
            ),
            _build_axis_rope(
                indexes[1], self.config.head_dim // 4, self.config.rope_theta_hw, dtype
            ),
            _build_axis_rope(
                indexes[2], self.config.head_dim // 4, self.config.rope_theta_hw, dtype
            ),
        )

    def forward_prefix(
        self,
        indexes: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        rope_cache = self.build_rope_cache(indexes, hidden_states.dtype)
        attention_runs = _prefix_attention_runs(indexes[0])
        cache = []
        residual = None
        for layer in self.layers:
            hidden_states, residual, layer_cache = layer(
                hidden_states,
                rope_cache=rope_cache,
                generation=False,
                attention_runs=attention_runs,
                residual=residual,
            )
            cache.append(layer_cache)
        return cache

    def forward_generation(
        self,
        inputs_embeds: torch.Tensor,
        indexes: torch.Tensor,
        prefix_cache: list[SenseNovaU1_5KVCache],
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        if rope_cache is None:
            rope_cache = self.build_rope_cache(indexes, hidden_states.dtype)
        residual = None
        for layer, layer_cache in zip(self.layers, prefix_cache):
            hidden_states, residual = layer(
                hidden_states,
                prefix_cache=layer_cache,
                rope_cache=rope_cache,
                generation=True,
                residual=residual,
            )
        normalized, _ = self.norm_mot_gen(hidden_states, residual)
        return normalized

    def forward_generation_branches(
        self,
        inputs_embeds: torch.Tensor,
        prefix_caches: list[list[SenseNovaU1_5KVCache]],
        rope_caches: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        residual = None
        for layer_index, layer in enumerate(self.layers):
            hidden_states, residual = layer.forward_generation_branches(
                hidden_states,
                [cache[layer_index] for cache in prefix_caches],
                rope_caches,
                residual,
            )
        normalized, _ = self.norm_mot_gen(hidden_states, residual)
        return normalized

    @staticmethod
    def prepare_generation_cache(
        prefix_cache: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        batch_size: int,
        generation_length: int,
    ) -> list[SenseNovaU1_5KVCache]:
        """Expand prefix KV once and reserve the per-step image-token region."""
        prepared = []
        for prefix_key, prefix_value in prefix_cache:
            prefix_length = prefix_key.shape[1]
            shape = (
                batch_size,
                prefix_length + generation_length,
                prefix_key.shape[2],
                prefix_key.shape[3],
            )
            key = prefix_key.new_empty(shape)
            value = prefix_value.new_empty(shape)
            key[:, :prefix_length].copy_(prefix_key)
            value[:, :prefix_length].copy_(prefix_value)
            prepared.append(SenseNovaU1_5KVCache(key, value, prefix_length))
        return prepared


class SenseNovaU1_5LanguageModel(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = SenseNovaU1_5Qwen3Model(config, quant_config)


def _build_vision_rope_cache(
    grid_sizes: Sequence[tuple[int, int]],
    embed_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the fixed 2-D vision RoPE tables for one request.

    The vision input geometry is constant throughout denoising.  Keeping the
    tables in FP32 preserves the reference implementation's FP32 rotation
    before its final cast back to the model dtype.
    """
    positions_x: list[torch.Tensor] = []
    positions_y: list[torch.Tensor] = []
    for height, width in grid_sizes:
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        positions_x.append(x.reshape(-1))
        positions_y.append(y.reshape(-1))

    half = embed_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half)
    )

    def sincos(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.outer(positions.float(), inv_freq)
        return frequencies.cos(), frequencies.sin()

    return (*sincos(torch.cat(positions_x)), *sincos(torch.cat(positions_y)))


def _apply_cached_vision_rope(
    x: torch.Tensor,
    rope_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos_x, sin_x, cos_y, sin_y = rope_cache
    half = x.shape[-1] // 2

    def interleaved(
        part: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        even, odd = part[..., 0::2], part[..., 1::2]
        out = torch.empty_like(part)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out

    return torch.cat(
        (
            interleaved(x[..., :half], cos_x, sin_x),
            interleaved(x[..., half:], cos_y, sin_y),
        ),
        dim=-1,
    )


class SenseNovaU1_5VisionEmbeddings(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.embed_dim = int(config.hidden_size)
        self.downsample_factor = int(1 / float(config.downsample_ratio))
        self.rope_theta = float(config.rope_theta_vision)
        self.patch_embedding = nn.Conv2d(
            int(config.num_channels),
            self.embed_dim,
            self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            self.embed_dim,
            int(config.llm_hidden_size),
            self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()

    def build_rope_cache(
        self, grid_sizes: Sequence[tuple[int, int]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _build_vision_rope_cache(
            grid_sizes, self.embed_dim, self.rope_theta, device
        )

    @staticmethod
    def _normalize_grid_sizes(
        grid_hw: torch.Tensor, grid_sizes: Sequence[tuple[int, int]] | None
    ) -> tuple[tuple[int, int], ...]:
        if grid_sizes is not None:
            return tuple((int(height), int(width)) for height, width in grid_sizes)
        # This fallback is used by one-time prefix encoding.  Generation
        # passes Python geometry explicitly and never synchronizes here.
        return tuple((int(height), int(width)) for height, width in grid_hw.tolist())

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        grid_sizes: Sequence[tuple[int, int]] | None = None,
        rope_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        grid_sizes = self._normalize_grid_sizes(grid_hw, grid_sizes)
        pixels = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patches = self.gelu(self.patch_embedding(pixels)).view(-1, self.embed_dim)
        if rope_cache is None:
            rope_cache = self.build_rope_cache(grid_sizes, patches.device)
        patches = _apply_cached_vision_rope(patches.float(), rope_cache).to(
            patches.dtype
        )
        outputs = []
        offset = 0
        for height, width in grid_sizes:
            count = height * width
            image = patches[offset : offset + count].view(1, height, width, -1)
            image = self.dense_embedding(image.permute(0, 3, 1, 2))
            outputs.append(image.permute(0, 2, 3, 1).reshape(-1, image.shape[1]))
            offset += count
        return torch.cat(outputs, dim=0)

    def forward_bchw(
        self,
        images: torch.Tensor,
        grid_sizes: Sequence[tuple[int, int]],
        rope_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        """Generation fast path that keeps the image in BCHW layout.

        The generation image has one fixed grid for every batch item, so the
        patch convolution can consume the image directly.  Editing prefixes
        may contain heterogeneous grids and continue to use ``forward``.
        """
        if len(grid_sizes) != images.shape[0]:
            raise ValueError(
                "grid_sizes must contain one entry per BCHW image, got "
                f"{len(grid_sizes)} for batch {images.shape[0]}"
            )
        expected = (
            images.shape[2] // self.patch_size,
            images.shape[3] // self.patch_size,
        )
        if any(tuple(size) != expected for size in grid_sizes):
            raise ValueError(
                "forward_bchw requires one common patch grid for all images"
            )

        patches = self.gelu(self.patch_embedding(images))
        patches = patches.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)
        if rope_cache is None:
            rope_cache = self.build_rope_cache(grid_sizes, patches.device)
        patches = _apply_cached_vision_rope(patches.float(), rope_cache).to(
            patches.dtype
        )
        height, width = expected
        patches = patches.view(images.shape[0], height, width, self.embed_dim)
        image = self.dense_embedding(patches.permute(0, 3, 1, 2))
        return image.permute(0, 2, 3, 1).reshape(-1, image.shape[1])


class SenseNovaU1_5VisionModel(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embeddings = SenseNovaU1_5VisionEmbeddings(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        grid_sizes: Sequence[tuple[int, int]] | None = None,
        rope_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        return self.embeddings(
            pixel_values, grid_hw, grid_sizes=grid_sizes, rope_cache=rope_cache
        )

    def forward_bchw(
        self,
        images: torch.Tensor,
        grid_sizes: Sequence[tuple[int, int]],
        rope_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        return self.embeddings.forward_bchw(images, grid_sizes, rope_cache)

    def build_rope_cache(
        self, grid_sizes: Sequence[tuple[int, int]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.embeddings.build_rope_cache(grid_sizes, device)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        # The model is constructed on ``meta`` during checkpoint loading.
        # A concrete non-checkpoint buffer created here would remain on meta
        # and fail the loader's post-load validation, so materialize this
        # deterministic cache lazily on the first real forward instead.
        self.register_buffer("_frequency_cache", None, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        freqs = self._frequency_cache
        if freqs is None or freqs.device != timestep.device:
            half = self.frequency_embedding_size // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=timestep.device, dtype=torch.float32)
                / half
            )
            self._frequency_cache = freqs
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, 3, padding=1)
        self.act1 = nn.GELU()
        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(hidden_dim // 4, 192, 3, padding=1)
        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps3(self.conv2(self.ps2(self.act1(self.conv1(self.ps1(x))))))


def _load_image_native(
    image: Image.Image,
    patch_size: int,
    downsample_ratio: float,
    min_pixels: int,
    max_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = image.convert("RGB")
    factor = int(patch_size / downsample_ratio)
    height, width = smart_resize(
        image.height, image.width, factor, min_pixels, max_pixels
    )
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    tensor = IMAGE_NORMALIZE(IMAGE_TO_TENSOR(image))
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = (
        tensor.view(3, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, 3 * patch_size**2)
    )
    return patches, torch.tensor([[grid_h, grid_w]], dtype=torch.long)


class SenseNovaU1_5NativeModel(nn.Module, LayerwiseOffloadableModuleMixin):
    """Complete non-thinking U1.5 model using SGLang-native Qwen3 layers."""

    param_names_mapping: dict = {
        r"^(language_model\.model\.layers\.\d+\.self_attn\.)q_proj\.(.*)$": (
            r"\1qkv_proj.\2",
            0,
            3,
        ),
        r"^(language_model\.model\.layers\.\d+\.self_attn\.)k_proj\.(.*)$": (
            r"\1qkv_proj.\2",
            1,
            3,
        ),
        r"^(language_model\.model\.layers\.\d+\.self_attn\.)v_proj\.(.*)$": (
            r"\1qkv_proj.\2",
            2,
            3,
        ),
        r"^(language_model\.model\.layers\.\d+\.self_attn\.)q_proj_mot_gen\.(.*)$": (
            r"\1qkv_proj_mot_gen.\2",
            0,
            3,
        ),
        r"^(language_model\.model\.layers\.\d+\.self_attn\.)k_proj_mot_gen\.(.*)$": (
            r"\1qkv_proj_mot_gen.\2",
            1,
            3,
        ),
        r"^(language_model\.model\.layers\.\d+\.self_attn\.)v_proj_mot_gen\.(.*)$": (
            r"\1qkv_proj_mot_gen.\2",
            2,
            3,
        ),
        r"^(language_model\.model\.layers\.\d+\.mlp\.)gate_proj\.(.*)$": (
            r"\1gate_up_proj.\2",
            0,
            2,
        ),
        r"^(language_model\.model\.layers\.\d+\.mlp\.)up_proj\.(.*)$": (
            r"\1gate_up_proj.\2",
            1,
            2,
        ),
        r"^(language_model\.model\.layers\.\d+\.mlp_mot_gen\.)gate_proj\.(.*)$": (
            r"\1gate_up_proj.\2",
            0,
            2,
        ),
        r"^(language_model\.model\.layers\.\d+\.mlp_mot_gen\.)up_proj\.(.*)$": (
            r"\1gate_up_proj.\2",
            1,
            2,
        ),
    }
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_proj_mot_gen": [
            "q_proj_mot_gen",
            "k_proj_mot_gen",
            "v_proj_mot_gen",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    layer_names = ["language_model.model.layers"]

    def __init__(
        self,
        config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        llm_values = dict(config["llm_config"])
        llm_values.setdefault(
            "head_dim", llm_values["hidden_size"] // llm_values["num_attention_heads"]
        )
        llm_values.setdefault("attention_bias", False)
        llm_values.setdefault("pad_token_id", int(config.get("pad_token_id", 151643)))
        llm_values.setdefault("rope_theta_hw", 10000.0)
        self.llm_config = _namespace(llm_values)

        vision_values = dict(config["vision_config"])
        vision_values["llm_hidden_size"] = self.llm_config.hidden_size
        vision_values["downsample_ratio"] = float(config.get("downsample_ratio", 0.5))
        self.vision_config = _namespace(vision_values)
        self.patch_size = int(config.get("patch_size", vision_values["patch_size"]))
        self.downsample_ratio = float(config.get("downsample_ratio", 0.5))
        self.language_model = SenseNovaU1_5LanguageModel(self.llm_config, quant_config)
        self.vision_model = SenseNovaU1_5VisionModel(self.vision_config)
        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": SenseNovaU1_5VisionModel(self.vision_config),
                "timestep_embedder": TimestepEmbedder(self.llm_config.hidden_size),
                "fm_head": ConvDecoder(self.llm_config.hidden_size),
            }
        )
        self.add_noise_scale_embedding = bool(
            config.get("add_noise_scale_embedding", False)
        )
        if self.add_noise_scale_embedding:
            self.fm_modules["noise_scale_embedder"] = TimestepEmbedder(
                self.llm_config.hidden_size
            )
        self.noise_scale = float(config.get("noise_scale", 1.0))
        self.noise_scale_mode = str(config.get("noise_scale_mode", "resolution"))
        self.noise_scale_base_image_seq_len = int(
            config.get("noise_scale_base_image_seq_len", 64)
        )
        self.noise_scale_max_value = float(config.get("noise_scale_max_value", 16.0))
        self.t_eps = float(config.get("t_eps", 0.02))
        self.img_start_token_id = 151670
        self.img_context_token_id: int | None = None

    @property
    def device(self) -> torch.device:
        return self.language_model.model.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.language_model.model.embed_tokens.weight.dtype

    @classmethod
    def config_from_path(cls, model_path: str | os.PathLike[str]) -> dict[str, Any]:
        with open(Path(model_path) / "config.json", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def should_materialize_checkpoint_weight(name: str) -> bool:
        return name != "language_model.lm_head.weight"

    def post_load_weights(self) -> None:
        return None

    @staticmethod
    def _query(prompt: str, *, system_message: str | None = None) -> str:
        if system_message is None:
            system_message = DEFAULT_SYSTEM_MESSAGE
        system = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            if system_message
            else ""
        )
        return f"{system}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def _text_indexes(length: int, device: torch.device) -> torch.Tensor:
        temporal = torch.arange(length, device=device, dtype=torch.long)
        return torch.stack(
            (temporal, torch.zeros_like(temporal), torch.zeros_like(temporal))
        )

    @staticmethod
    def _image_indexes(
        token_h: int, token_w: int, text_len: int, device: torch.device
    ) -> torch.Tensor:
        flat = torch.arange(token_h * token_w, device=device)
        return torch.stack(
            (
                torch.full_like(flat, text_len),
                flat // token_w,
                flat % token_w,
            )
        )

    def _thw_indexes(
        self, input_ids: torch.Tensor, grid_hw: torch.Tensor | None
    ) -> torch.Tensor:
        if self.img_context_token_id is None:
            raise RuntimeError("img_context_token_id is not initialized")
        start_shift = torch.cat(
            (
                torch.zeros(1, device=input_ids.device, dtype=torch.long),
                (input_ids == self.img_start_token_id).long(),
            )
        )[:-1]
        not_image = (input_ids != self.img_context_token_id).long()
        temporal = (start_shift + not_image).cumsum(0) - 1
        height = torch.zeros_like(temporal)
        width = torch.zeros_like(temporal)
        if grid_hw is not None:
            selected = input_ids == self.img_context_token_id
            positions_h, positions_w = [], []
            merge = int(1 / self.downsample_ratio)
            for grid_h, grid_w in (grid_hw // merge).tolist():
                y, x = torch.meshgrid(
                    torch.arange(grid_h, device=input_ids.device),
                    torch.arange(grid_w, device=input_ids.device),
                    indexing="ij",
                )
                positions_h.append(y.reshape(-1))
                positions_w.append(x.reshape(-1))
            height[selected] = torch.cat(positions_h)
            width[selected] = torch.cat(positions_w)
        return torch.stack((temporal, height, width))

    @staticmethod
    def _patchify(images: torch.Tensor, patch_size: int, channel_first: bool = False):
        h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
        x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
        permutation = (0, 2, 4, 1, 3, 5) if channel_first else (0, 2, 4, 3, 5, 1)
        return x.permute(permutation).reshape(images.shape[0], h * w, patch_size**2 * 3)

    @staticmethod
    def _unpatchify(x: torch.Tensor, patch_size: int, height: int, width: int):
        h, w = height // patch_size, width // patch_size
        x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], 3, height, width)

    @staticmethod
    def _shift_timesteps(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
        sigma = 1 - timesteps
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
        return 1 - sigma

    def _predict_velocity_branches(
        self,
        image_embeds: torch.Tensor,
        branches: list[
            tuple[
                torch.Tensor,
                list[SenseNovaU1_5KVCache],
                tuple[tuple[torch.Tensor, torch.Tensor], ...],
            ]
        ],
        z: torch.Tensor | None,
        timestep: torch.Tensor,
        image_size: tuple[int, int],
        *,
        image_bchw: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        branch_count = len(branches)
        model_inputs = (
            image_embeds
            if branch_count == 1
            else image_embeds.repeat(branch_count, 1, 1)
        )
        hidden = self.language_model.model.forward_generation_branches(
            model_inputs,
            [branch[1] for branch in branches],
            [branch[2] for branch in branches],
        )
        if image_bchw is not None:
            batch = image_bchw.shape[0]
            merge = int(1 / self.downsample_ratio)
            token_h = image_size[1] // (self.patch_size * merge)
            token_w = image_size[0] // (self.patch_size * merge)
            image_2d = hidden.view(branch_count * batch, token_h, token_w, -1).permute(
                0, 3, 1, 2
            )
            decoded = self.fm_modules["fm_head"](image_2d)
            velocity = (
                decoded.view(
                    branch_count, batch, 3, image_bchw.shape[2], image_bchw.shape[3]
                )
                - image_bchw.unsqueeze(0)
            ) / (1 - timestep).clamp_min(self.t_eps)
            return list(velocity.unbind(0))

        if z is None:
            raise ValueError("z is required for the patch-token velocity path")
        batch, length = z.shape[:2]
        merge = int(1 / self.downsample_ratio)
        token_h = image_size[1] // (self.patch_size * merge)
        token_w = image_size[0] // (self.patch_size * merge)
        image_2d = hidden.view(branch_count * batch, token_h, token_w, -1).permute(
            0, 3, 1, 2
        )
        decoded = self.fm_modules["fm_head"](image_2d)
        decoded = decoded.view(
            branch_count * batch,
            3,
            token_h,
            self.patch_size * merge,
            token_w,
            self.patch_size * merge,
        )
        predicted = decoded.permute(0, 2, 4, 3, 5, 1).reshape(
            branch_count * batch, length, (self.patch_size * merge) ** 2 * 3
        )
        z_inputs = z if branch_count == 1 else z.repeat(branch_count, 1, 1)
        velocity = (predicted - z_inputs) / (1 - timestep).clamp_min(self.t_eps)
        return list(velocity.view(branch_count, batch, length, -1).unbind(0))

    def _initial_noise(
        self, batch_size: int, image_size: tuple[int, int], seed: int
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        merge = int(1 / self.downsample_ratio)
        grid_h, grid_w = (
            image_size[1] // self.patch_size,
            image_size[0] // self.patch_size,
        )
        noise_scale = self.noise_scale
        if self.noise_scale_mode in {"resolution", "dynamic", "dynamic_sqrt"}:
            scale = math.sqrt(
                (grid_h * grid_w)
                / merge**2
                / float(self.noise_scale_base_image_seq_len)
            )
            noise_scale *= scale
            if self.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)
        generator = torch.Generator(self.device).manual_seed(seed)
        image = noise_scale * torch.randn(
            (batch_size, 3, image_size[1], image_size[0]),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        grid_hw = torch.tensor([[grid_h, grid_w]] * batch_size, device=self.device)
        return image, noise_scale, grid_hw

    def _denoise(
        self,
        condition_cache: list[tuple[torch.Tensor, torch.Tensor]],
        condition_indexes: torch.Tensor,
        image_size: tuple[int, int],
        num_steps: int,
        batch_size: int,
        seed: int,
        timestep_shift: float,
        cfg_scale: float,
        cfg_norm: str,
        uncondition_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        uncondition_indexes: torch.Tensor | None = None,
        img_condition_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        img_condition_indexes: torch.Tensor | None = None,
        img_cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        image, noise_scale, grid_hw = self._initial_noise(batch_size, image_size, seed)
        merge = int(1 / self.downsample_ratio)
        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        token_h, token_w = grid_h // merge, grid_w // merge
        generation_length = token_h * token_w
        condition_cache = self.language_model.model.prepare_generation_cache(
            condition_cache,
            batch_size=batch_size,
            generation_length=generation_length,
        )
        if uncondition_cache is not None:
            uncondition_cache = self.language_model.model.prepare_generation_cache(
                uncondition_cache,
                batch_size=batch_size,
                generation_length=generation_length,
            )
        if img_condition_cache is not None:
            img_condition_cache = self.language_model.model.prepare_generation_cache(
                img_condition_cache,
                batch_size=batch_size,
                generation_length=generation_length,
            )
        timesteps = self._shift_timesteps(
            torch.linspace(0, 1, num_steps + 1, device=self.device), timestep_shift
        )
        timestep_embeddings = self.fm_modules["timestep_embedder"](timesteps[:-1]).view(
            num_steps, 1, 1, -1
        )
        if self.add_noise_scale_embedding:
            noise_scale_embedding = self.fm_modules["noise_scale_embedder"](
                torch.tensor(
                    [noise_scale / self.noise_scale_max_value],
                    device=self.device,
                    dtype=torch.float32,
                )
            ).view(1, 1, -1)
            timestep_embeddings = timestep_embeddings + noise_scale_embedding

        vision_grid_sizes = ((grid_h, grid_w),) * batch_size
        vision_rope_cache = self.fm_modules["vision_model_mot_gen"].build_rope_cache(
            vision_grid_sizes, self.device
        )
        use_bchw = cfg_norm == "none"
        rope_caches = {}
        for indexes in (condition_indexes, img_condition_indexes, uncondition_indexes):
            if indexes is not None and id(indexes) not in rope_caches:
                rope_caches[id(indexes)] = self.language_model.model.build_rope_cache(
                    indexes, self.dtype
                )
        for step in range(num_steps):
            timestep, next_timestep = timesteps[step], timesteps[step + 1]
            z = None if use_bchw else self._patchify(image, self.patch_size * merge)
            if use_bchw:
                embeds = (
                    self.fm_modules["vision_model_mot_gen"]
                    .forward_bchw(image, vision_grid_sizes, vision_rope_cache)
                    .view(batch_size, token_h * token_w, -1)
                )
            else:
                image_input = self._patchify(image, self.patch_size, channel_first=True)
                embeds = self.fm_modules["vision_model_mot_gen"](
                    image_input.view(batch_size * grid_h * grid_w, -1),
                    grid_hw,
                    grid_sizes=vision_grid_sizes,
                    rope_cache=vision_rope_cache,
                ).view(batch_size, token_h * token_w, -1)
            embeds = embeds + timestep_embeddings[step]
            branches = [(condition_indexes, condition_cache)]
            if uncondition_cache is None and img_condition_cache is None:
                branch_names = ("condition",)
            elif img_condition_cache is None:
                branches.append((uncondition_indexes, uncondition_cache))
                branch_names = ("condition", "uncondition")
            elif uncondition_cache is None:
                branches.append((img_condition_indexes, img_condition_cache))
                branch_names = ("condition", "image")
            elif cfg_scale == 1 and img_cfg_scale == 1:
                branch_names = ("condition",)
            elif img_cfg_scale == 1:
                branches.append((img_condition_indexes, img_condition_cache))
                branch_names = ("condition", "image")
            elif cfg_scale == img_cfg_scale:
                branches.append((uncondition_indexes, uncondition_cache))
                branch_names = ("condition", "uncondition")
            else:
                branches.extend(
                    [
                        (img_condition_indexes, img_condition_cache),
                        (uncondition_indexes, uncondition_cache),
                    ]
                )
                branch_names = ("condition", "image", "uncondition")
            branch_outputs = self._predict_velocity_branches(
                embeds,
                [
                    (indexes, cache, rope_caches[id(indexes)])
                    for indexes, cache in branches
                ],
                z,
                timestep,
                image_size,
                image_bchw=image if use_bchw else None,
            )
            condition = branch_outputs[0]
            if branch_names == ("condition",):
                velocity = condition
            elif len(branch_outputs) == 2:
                velocity = branch_outputs[1] + cfg_scale * (
                    condition - branch_outputs[1]
                )
            else:
                image_condition, uncondition = branch_outputs[1:]
                velocity = (
                    uncondition
                    + cfg_scale * (condition - image_condition)
                    + img_cfg_scale * (image_condition - uncondition)
                )
            if cfg_norm == "global" and (cfg_scale > 1 or img_cfg_scale > 1):
                scale = (
                    torch.norm(condition, dim=(1, 2), keepdim=True)
                    / (torch.norm(velocity, dim=(1, 2), keepdim=True) + 1e-8)
                ).clamp(max=1)
                velocity = velocity * scale
            elif cfg_norm == "channel" and (cfg_scale > 1 or img_cfg_scale > 1):
                scale = (
                    torch.norm(condition, dim=-1, keepdim=True)
                    / (torch.norm(velocity, dim=-1, keepdim=True) + 1e-8)
                ).clamp(max=1)
                velocity = velocity * scale
            if use_bchw:
                image = image + (next_timestep - timestep) * velocity
            else:
                image = self._unpatchify(
                    z + (next_timestep - timestep) * velocity,
                    self.patch_size * merge,
                    image_size[1],
                    image_size[0],
                )
        return image

    @torch.inference_mode()
    def t2i_generate(
        self,
        tokenizer,
        prompt: str,
        cfg_scale: float = 4.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        t_eps: float = 0.02,
        image_size: tuple[int, int] = (2048, 2048),
        num_steps: int = 50,
        batch_size: int = 1,
        seed: int = 0,
        **_: Any,
    ) -> torch.Tensor:
        self.t_eps = float(t_eps)
        suffix = "<think>\n\n</think>\n\n<img>"
        query = self._query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN) + suffix
        ids = tokenizer(query, return_tensors="pt")["input_ids"].to(self.device)
        indexes = self._text_indexes(ids.shape[1], self.device)
        condition_cache = self.language_model.model.forward_prefix(
            indexes, input_ids=ids
        )
        merge = int(1 / self.downsample_ratio)
        token_h = image_size[1] // (self.patch_size * merge)
        token_w = image_size[0] // (self.patch_size * merge)
        condition_image_indexes = self._image_indexes(
            token_h, token_w, ids.shape[1], self.device
        )
        uncondition_cache = uncondition_indexes = None
        if cfg_scale > 1:
            uncond_query = self._query("") + "<img>"
            uncond_ids = tokenizer(uncond_query, return_tensors="pt")["input_ids"].to(
                self.device
            )
            uncond_text_indexes = self._text_indexes(uncond_ids.shape[1], self.device)
            uncondition_cache = self.language_model.model.forward_prefix(
                uncond_text_indexes, input_ids=uncond_ids
            )
            uncondition_indexes = self._image_indexes(
                token_h, token_w, uncond_ids.shape[1], self.device
            )
        return self._denoise(
            condition_cache,
            condition_image_indexes,
            image_size,
            num_steps,
            batch_size,
            seed,
            timestep_shift,
            cfg_scale,
            cfg_norm,
            uncondition_cache,
            uncondition_indexes,
        )

    def _editing_prefix(
        self,
        tokenizer,
        query: str,
        pixel_values: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
        visual_embeds: torch.Tensor | None = None,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        ids = tokenizer(query, return_tensors="pt")["input_ids"].to(self.device)
        indexes = self._thw_indexes(ids[0], grid_hw)
        embeds = self.language_model.model.embed_tokens(ids)
        if pixel_values is not None:
            if visual_embeds is None:
                visual_embeds = self.vision_model(pixel_values, grid_hw).to(
                    embeds.dtype
                )
            elif visual_embeds.dtype != embeds.dtype:
                visual_embeds = visual_embeds.to(embeds.dtype)
            selected = ids[0] == self.img_context_token_id
            embeds[0, selected] = visual_embeds
        cache = self.language_model.model.forward_prefix(indexes, inputs_embeds=embeds)
        return cache, int(indexes[0].max().item()) + 1

    @torch.inference_mode()
    def it2i_generate(
        self,
        tokenizer,
        prompt: str,
        images: list[Image.Image],
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        t_eps: float = 0.02,
        image_size: tuple[int, int] = (2048, 2048),
        num_steps: int = 50,
        batch_size: int = 1,
        seed: int = 0,
        **_: Any,
    ) -> torch.Tensor:
        self.t_eps = float(t_eps)
        self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
        image_count = prompt.count("<image>")
        if image_count > len(images):
            raise ValueError(
                f"prompt contains {image_count} <image> placeholders, but only "
                f"{len(images)} reference images were supplied"
            )
        if image_count == 0:
            if len(images) > 1:
                prompt = (
                    "".join(
                        f"Image-{index + 1}:<image>\n" for index in range(len(images))
                    )
                    + prompt
                )
            else:
                prompt = "<image>\n" + prompt
        elif image_count < len(images):
            prompt = "<image>\n" * (len(images) - image_count) + prompt
        pixel_values, grids = [], []
        for image in images:
            pixels, grid = _load_image_native(
                image,
                self.patch_size,
                self.downsample_ratio,
                512 * 512,
                min(2048 * 2048, (4096 * 4096) // len(images)),
            )
            pixel_values.append(pixels)
            grids.append(grid)
        pixel_values = torch.cat(pixel_values).to(self.device, self.dtype)
        grid_hw = torch.cat(grids).to(self.device)
        merge = int(1 / self.downsample_ratio)
        visual_embeds = self.vision_model(pixel_values, grid_hw).to(self.dtype)

        def inject_images(query: str) -> str:
            for grid_h, grid_w in grid_hw.tolist():
                count = int(grid_h * grid_w / merge**2)
                tokens = "<img>" + "<IMG_CONTEXT>" * count + "</img>"
                query = query.replace("<image>", tokens, 1)
            return query

        suffix = "<think>\n\n</think>\n\n<img>"
        condition_query = inject_images(
            self._query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN) + suffix
        )
        condition_cache, condition_len = self._editing_prefix(
            tokenizer, condition_query, pixel_values, grid_hw, visual_embeds
        )
        token_h = image_size[1] // (self.patch_size * merge)
        token_w = image_size[0] // (self.patch_size * merge)
        condition_indexes = self._image_indexes(
            token_h, token_w, condition_len, self.device
        )

        needs_cfg = not (cfg_scale == 1 and img_cfg_scale == 1)
        needs_img_condition = needs_cfg and (
            img_cfg_scale == 1 or cfg_scale != img_cfg_scale
        )
        needs_uncondition = needs_cfg and img_cfg_scale != 1
        image_cache = image_indexes = uncondition_cache = uncondition_indexes = None
        if needs_img_condition:
            image_query = inject_images(self._query("<image>" * len(images)) + "<img>")
            image_cache, image_len = self._editing_prefix(
                tokenizer, image_query, pixel_values, grid_hw, visual_embeds
            )
            image_indexes = self._image_indexes(
                token_h, token_w, image_len, self.device
            )
        if needs_uncondition:
            uncondition_cache, uncondition_len = self._editing_prefix(
                tokenizer, self._query("") + "<img>"
            )
            uncondition_indexes = self._image_indexes(
                token_h, token_w, uncondition_len, self.device
            )
        return self._denoise(
            condition_cache,
            condition_indexes,
            image_size,
            num_steps,
            batch_size,
            seed,
            timestep_shift,
            cfg_scale,
            cfg_norm,
            uncondition_cache,
            uncondition_indexes,
            image_cache,
            image_indexes,
            img_cfg_scale,
        )


EntryClass = SenseNovaU1_5NativeModel

__all__ = ["SenseNovaU1_5NativeModel"]
