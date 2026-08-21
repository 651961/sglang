# SPDX-License-Identifier: Apache-2.0
"""Native SenseNova-U1.5 Qwen3/MoT image generator."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from sglang.kernels.ops.diffusion import (
    can_use_fused_silu_mul,
    can_use_fused_rope_rotate_half,
    fused_silu_mul_bitexact,
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
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import (
    Qwen3VLRowParallelLinear,
    _gather_tensor_parallel_activation,
    _make_text_row_linear,
    _tp_world_size,
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
        batch = q.shape[0]
        cos_rows = cos.expand(batch, -1, -1).reshape(-1, cos.shape[-1])
        sin_rows = sin.expand(batch, -1, -1).reshape(-1, sin.shape[-1])
        can_fuse = can_use_fused_rope_rotate_half(q, cos_rows, sin_rows)
        can_fuse = can_fuse and can_use_fused_rope_rotate_half(
            k, cos_rows, sin_rows
        )
        if can_fuse:
            return (
                fused_rope_rotate_half_bitexact(q, cos_rows, sin_rows),
                fused_rope_rotate_half_bitexact(k, cos_rows, sin_rows),
            )
    cos = cos[None, :, None]
    sin = sin[None, :, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _block_causal_mask(block_index: torch.Tensor) -> torch.Tensor:
    """Build the official U1.5 ``same_block | causal`` prefix mask."""
    length = block_index.numel()
    row_blocks = block_index[:, None].expand(length, length)
    col_blocks = block_index[None, :].expand(length, length)
    positions = torch.arange(length, device=block_index.device)
    allowed = (col_blocks == row_blocks) | (
        positions[None, :] <= positions[:, None]
    )
    return torch.where(
        allowed[None, None],
        torch.zeros((), device=block_index.device),
        torch.full((), float("-inf"), device=block_index.device),
    )


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
        tp_size = _tp_world_size()
        use_tp = (
            tp_size > 1
            and config.num_attention_heads % tp_size == 0
            and config.num_key_value_heads % tp_size == 0
        )
        self.tp_size = tp_size if use_tp else 1
        self.num_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = config.num_key_value_heads // self.tp_size
        self.scaling = self.head_dim**-0.5

        q_out = config.num_attention_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=bool(config.attention_bias),
            quant_config=quant_config,
            prefix=(
                f"language_model.model.layers.{layer_idx}.self_attn.qkv_proj"
            ),
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

        def row(suffix: str):
            if quant_config is not None:
                return _make_sensenova_linear(
                    q_out,
                    config.hidden_size,
                    bias=bool(config.attention_bias),
                    quant_config=quant_config,
                    prefix=(
                        f"language_model.model.layers.{layer_idx}."
                        f"self_attn.{suffix}"
                    ),
                )
            return _make_text_row_linear(
                q_out,
                config.hidden_size,
                bias=bool(config.attention_bias),
                quant_config=None,
                use_weight_only_fp8=False,
                use_tensor_parallel=use_tp,
                prefix=f"layers.{layer_idx}.self_attn.{suffix}",
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
        indexes: torch.Tensor,
        generation: bool,
        rope_cache: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if generation:
            q_proj = self.qkv_proj_mot_gen
            q_norm, k_norm = self.q_norm_mot_gen, self.k_norm_mot_gen
            q_norm_hw, k_norm_hw = self.q_norm_hw_mot_gen, self.k_norm_hw_mot_gen
        else:
            q_proj = self.qkv_proj
            q_norm, k_norm = self.q_norm, self.k_norm
            q_norm_hw, k_norm_hw = self.q_norm_hw, self.k_norm_hw

        batch, seq_len, _ = hidden_states.shape
        qkv = _linear_output(q_proj, hidden_states)
        q, k, v = qkv.split(
            [self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim,
             self.num_kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q_t, q_hw = q.chunk(2, dim=-1)
        k_t, k_hw = k.chunk(2, dim=-1)
        q_t = q_norm(q_t)
        k_t = k_norm(k_t)
        q_hw = q_norm_hw(q_hw)
        k_hw = k_norm_hw(k_hw)
        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        q_t, k_t = _apply_axis_rope(q_t, k_t, rope_cache[0])
        q_h, k_h = _apply_axis_rope(q_h, k_h, rope_cache[1])
        q_w, k_w = _apply_axis_rope(q_w, k_w, rope_cache[2])
        q = torch.cat((q_t, q_h, q_w), dim=-1).contiguous()
        k = torch.cat((k_t, k_h, k_w), dim=-1).contiguous()
        return q, k, v

    def _output_projection(
        self, output: torch.Tensor, generation: bool
    ) -> torch.Tensor:
        output = output.reshape(*output.shape[:2], -1).contiguous()
        projection = self.o_proj_mot_gen if generation else self.o_proj
        if not isinstance(projection, Qwen3VLRowParallelLinear):
            output = _gather_tensor_parallel_activation(output, projection)
        return _linear_output(projection, output)

    def forward_prefix(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self._qkv(hidden_states, indexes, generation=False, rope_cache=rope_cache)
        output = self.prefix_attention(q, k, v, attn_mask=attn_mask)
        return self._output_projection(output, generation=False), (
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )

    def forward_generation(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        prefix_cache: tuple[torch.Tensor, torch.Tensor],
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        q, k, v = self._qkv(hidden_states, indexes, generation=True, rope_cache=rope_cache)
        prefix_k, prefix_v = prefix_cache
        batch = hidden_states.shape[0]
        prefix_k = prefix_k.expand(batch, -1, -1, -1)
        prefix_v = prefix_v.expand(batch, -1, -1, -1)
        k = torch.cat((prefix_k.transpose(1, 2), k), dim=1)
        v = torch.cat((prefix_v.transpose(1, 2), v), dim=1)
        output = self.generation_attention(q, k, v)
        return self._output_projection(output, generation=True)


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
        gate, up = gate_up.chunk(2, dim=-1)
        if can_use_fused_silu_mul(gate, up):
            hidden_states = fused_silu_mul_bitexact(gate, up)
        else:
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
        self.mlp = SenseNovaU1_5MLP(
            config, quant_config, prefix=f"{prefix}.mlp"
        )
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
        indexes: torch.Tensor,
        *,
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        generation: bool,
        attn_mask: torch.Tensor | None = None,
        prefix_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        if generation:
            if prefix_cache is None:
                raise ValueError("generation requires prefix_cache")
            return self.forward_generation(
                hidden_states, indexes, prefix_cache, rope_cache
            )
        if attn_mask is None:
            raise ValueError("prefix requires attn_mask")
        return self.forward_prefix(hidden_states, indexes, attn_mask, rope_cache)

    def forward_prefix(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        attention, cache = self.self_attn.forward_prefix(
            self.input_layernorm(hidden_states), indexes, attn_mask, rope_cache
        )
        hidden_states = residual + attention
        return hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        ), cache

    def forward_generation(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        prefix_cache: tuple[torch.Tensor, torch.Tensor],
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        residual = hidden_states
        attention = self.self_attn.forward_generation(
            self.input_layernorm_mot_gen(hidden_states), indexes, prefix_cache, rope_cache
        )
        hidden_states = residual + attention
        return hidden_states + self.mlp_mot_gen(
            self.post_attention_layernorm_mot_gen(hidden_states)
        )


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
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        rope_cache = self.build_rope_cache(indexes, hidden_states.dtype)
        attn_mask = _block_causal_mask(indexes[0])
        cache = []
        for layer in self.layers:
            hidden_states, layer_cache = layer(
                hidden_states,
                indexes,
                attn_mask=attn_mask,
                rope_cache=rope_cache,
                generation=False,
            )
            cache.append(layer_cache)
        return cache, self.norm(hidden_states)

    def forward_generation(
        self,
        inputs_embeds: torch.Tensor,
        indexes: torch.Tensor,
        prefix_cache: list[tuple[torch.Tensor, torch.Tensor]],
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        if rope_cache is None:
            rope_cache = self.build_rope_cache(indexes, hidden_states.dtype)
        for layer, layer_cache in zip(self.layers, prefix_cache):
            hidden_states = layer(
                hidden_states,
                indexes,
                prefix_cache=layer_cache,
                rope_cache=rope_cache,
                generation=True,
            )
        return self.norm_mot_gen(hidden_states)


class SenseNovaU1_5LanguageModel(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = SenseNovaU1_5Qwen3Model(config, quant_config)


def _vision_rope(
    x: torch.Tensor, positions: torch.Tensor, theta: float
) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_part, y_part = x[..., :half], x[..., half:]

    def interleaved(part: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        dim = part.shape[-1]
        inv = 1.0 / (
            theta
            ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
        )
        freq = torch.outer(pos.float(), inv)
        cos, sin = freq.cos().to(x.dtype), freq.sin().to(x.dtype)
        even, odd = part[..., 0::2], part[..., 1::2]
        out = torch.empty_like(part)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out

    return torch.cat(
        (interleaved(x_part, positions[0]), interleaved(y_part, positions[1])), -1
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

    def forward(
        self, pixel_values: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        pixels = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patches = self.gelu(self.patch_embedding(pixels)).view(-1, self.embed_dim)
        positions = []
        for height, width in grid_hw.tolist():
            y, x = torch.meshgrid(
                torch.arange(height, device=patches.device),
                torch.arange(width, device=patches.device),
                indexing="ij",
            )
            positions.append(torch.stack((x.reshape(-1), y.reshape(-1))))
        patches = _vision_rope(
            patches.float(), torch.cat(positions, dim=1), self.rope_theta
        ).to(patches.dtype)
        outputs = []
        offset = 0
        for height, width in grid_hw.tolist():
            count = height * width
            image = patches[offset : offset + count].view(1, height, width, -1)
            image = self.dense_embedding(image.permute(0, 3, 1, 2))
            outputs.append(image.permute(0, 2, 3, 1).reshape(-1, image.shape[1]))
            offset += count
        return torch.cat(outputs, dim=0)


class SenseNovaU1_5VisionModel(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embeddings = SenseNovaU1_5VisionEmbeddings(config)

    def forward(
        self, pixel_values: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        return self.embeddings(pixel_values, grid_hw)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
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
    lora_param_names_mapping: dict = {}
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
        llm_values.setdefault("attention_dropout", 0.0)
        llm_values.setdefault("hidden_act", "silu")
        llm_values.setdefault("pad_token_id", int(config.get("pad_token_id", 151643)))
        llm_values.setdefault("rope_theta_hw", 10000.0)
        self.llm_config = _namespace(llm_values)

        vision_values = dict(config["vision_config"])
        vision_values["llm_hidden_size"] = self.llm_config.hidden_size
        vision_values["downsample_ratio"] = float(config.get("downsample_ratio", 0.5))
        self.vision_config = _namespace(vision_values)
        self.patch_size = int(config.get("patch_size", vision_values["patch_size"]))
        self.downsample_ratio = float(config.get("downsample_ratio", 0.5))
        self.language_model = SenseNovaU1_5LanguageModel(
            self.llm_config, quant_config
        )
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
        return x.permute(permutation).reshape(
            images.shape[0], h * w, patch_size**2 * 3
        )

    @staticmethod
    def _unpatchify(x: torch.Tensor, patch_size: int, height: int, width: int):
        h, w = height // patch_size, width // patch_size
        x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(
            x.shape[0], 3, height, width
        )

    @staticmethod
    def _shift_timesteps(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
        sigma = 1 - timesteps
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
        return 1 - sigma

    def _predict_velocity(
        self,
        image_embeds: torch.Tensor,
        indexes: torch.Tensor,
        prefix_cache: list[tuple[torch.Tensor, torch.Tensor]],
        rope_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        z: torch.Tensor,
        timestep: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        hidden = self.language_model.model.forward_generation(
            image_embeds, indexes, prefix_cache, rope_cache
        )
        batch, length = z.shape[:2]
        merge = int(1 / self.downsample_ratio)
        token_h = image_size[1] // (self.patch_size * merge)
        token_w = image_size[0] // (self.patch_size * merge)
        image_2d = hidden.view(batch, token_h, token_w, -1).permute(0, 3, 1, 2)
        decoded = self.fm_modules["fm_head"](image_2d)
        decoded = decoded.view(
            batch,
            3,
            token_h,
            self.patch_size * merge,
            token_w,
            self.patch_size * merge,
        )
        predicted = decoded.permute(0, 2, 4, 3, 5, 1).reshape(
            batch, length, (self.patch_size * merge) ** 2 * 3
        )
        return (predicted - z) / (1 - timestep).clamp_min(self.t_eps)

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
        token_h, token_w = int(grid_hw[0, 0]) // merge, int(grid_hw[0, 1]) // merge
        timesteps = self._shift_timesteps(
            torch.linspace(0, 1, num_steps + 1, device=self.device), timestep_shift
        )
        rope_caches = {}
        for indexes in (condition_indexes, img_condition_indexes, uncondition_indexes):
            if indexes is not None and id(indexes) not in rope_caches:
                rope_caches[id(indexes)] = self.language_model.model.build_rope_cache(
                    indexes, self.dtype
                )
        for step in range(num_steps):
            timestep, next_timestep = timesteps[step], timesteps[step + 1]
            z = self._patchify(image, self.patch_size * merge)
            image_input = self._patchify(image, self.patch_size, channel_first=True)
            embeds = self.fm_modules["vision_model_mot_gen"](
                image_input.view(
                    batch_size * int(grid_hw[0, 0]) * int(grid_hw[0, 1]), -1
                ),
                grid_hw,
            ).view(batch_size, token_h * token_w, -1)
            expanded = timestep.expand(batch_size * token_h * token_w)
            time_embed = self.fm_modules["timestep_embedder"](expanded).view_as(embeds)
            if self.add_noise_scale_embedding:
                noise_value = torch.full_like(
                    expanded, noise_scale / self.noise_scale_max_value
                )
                time_embed = time_embed + self.fm_modules["noise_scale_embedder"](
                    noise_value
                ).view_as(embeds)
            embeds = embeds + time_embed
            condition = self._predict_velocity(
                embeds,
                condition_indexes,
                condition_cache,
                rope_caches[id(condition_indexes)],
                z,
                timestep,
                image_size,
            )
            if uncondition_cache is None and img_condition_cache is None:
                velocity = condition
            elif img_condition_cache is None:
                uncondition = self._predict_velocity(
                    embeds,
                    uncondition_indexes,
                    uncondition_cache,
                    rope_caches[id(uncondition_indexes)],
                    z,
                    timestep,
                    image_size,
                )
                velocity = uncondition + cfg_scale * (condition - uncondition)
            elif uncondition_cache is None:
                image_condition = self._predict_velocity(
                    embeds,
                    img_condition_indexes,
                    img_condition_cache,
                    rope_caches[id(img_condition_indexes)],
                    z,
                    timestep,
                    image_size,
                )
                velocity = image_condition + cfg_scale * (condition - image_condition)
            elif cfg_scale == 1 and img_cfg_scale == 1:
                velocity = condition
            elif img_cfg_scale == 1:
                image_condition = self._predict_velocity(
                    embeds,
                    img_condition_indexes,
                    img_condition_cache,
                    rope_caches[id(img_condition_indexes)],
                    z,
                    timestep,
                    image_size,
                )
                velocity = image_condition + cfg_scale * (condition - image_condition)
            elif cfg_scale == img_cfg_scale:
                uncondition = self._predict_velocity(
                    embeds,
                    uncondition_indexes,
                    uncondition_cache,
                    rope_caches[id(uncondition_indexes)],
                    z,
                    timestep,
                    image_size,
                )
                velocity = uncondition + cfg_scale * (condition - uncondition)
            else:
                image_condition = self._predict_velocity(
                    embeds,
                    img_condition_indexes,
                    img_condition_cache,
                    rope_caches[id(img_condition_indexes)],
                    z,
                    timestep,
                    image_size,
                )
                uncondition = self._predict_velocity(
                    embeds,
                    uncondition_indexes,
                    uncondition_cache,
                    rope_caches[id(uncondition_indexes)],
                    z,
                    timestep,
                    image_size,
                )
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
        image_size: tuple[int, int] = (2048, 2048),
        num_steps: int = 50,
        batch_size: int = 1,
        think_mode: bool = False,
        seed: int = 0,
        **_: Any,
    ) -> torch.Tensor:
        if think_mode:
            raise NotImplementedError("SenseNova-U1.5 thinking mode is not implemented")
        suffix = "<think>\n\n</think>\n\n<img>"
        query = self._query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN) + suffix
        ids = tokenizer(query, return_tensors="pt")["input_ids"].to(self.device)
        indexes = self._text_indexes(ids.shape[1], self.device)
        condition_cache, _ = self.language_model.model.forward_prefix(
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
            uncondition_cache, _ = self.language_model.model.forward_prefix(
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
                visual_embeds = self.vision_model(pixel_values, grid_hw).to(embeds.dtype)
            elif visual_embeds.dtype != embeds.dtype:
                visual_embeds = visual_embeds.to(embeds.dtype)
            selected = ids[0] == self.img_context_token_id
            embeds[0, selected] = visual_embeds
        cache, _ = self.language_model.model.forward_prefix(
            indexes, inputs_embeds=embeds
        )
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
        image_size: tuple[int, int] = (2048, 2048),
        num_steps: int = 50,
        batch_size: int = 1,
        think_mode: bool = False,
        seed: int = 0,
        **_: Any,
    ) -> torch.Tensor:
        if think_mode:
            raise NotImplementedError("SenseNova-U1.5 thinking mode is not implemented")
        self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
        image_count = prompt.count("<image>")
        if image_count == 0:
            if len(images) > 1:
                prompt = "".join(
                    f"Image-{index + 1}:<image>\n" for index in range(len(images))
                ) + prompt
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
