# SPDX-License-Identifier: Apache-2.0
"""Spark-H3 attention over reference-video and target-video rows (joint-video).

The optional Spark-H3 package owns Reblock, Top-K and Reweight kernels. This
adapter preserves H3's packed document boundaries and leaves images, keyframes,
text and audio dense. Ulysses exchanges run in the H3 attention core as usual.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class SparkAttentionMetadata(AttentionMetadata):
    num_steps: int
    used: int
    seq_len: int
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    video_positions: torch.Tensor
    video_tokens: int
    grid: tuple[int, int, int]
    # One controller per request, shared across DiT layers, released with it.
    controller: Any = field(default=None, repr=False)


class SparkAttentionMetadataBuilder(AttentionMetadataBuilder):
    def build(
        self, *, packed: dict[str, Any], num_steps: int, device: torch.device
    ) -> SparkAttentionMetadata:
        stream = packed["stream_layout"]
        img_pos = packed["img_pos"].cpu().view(-1)
        update = packed["update_mask"].cpu().view(-1)
        conditions = img_pos[~update]
        shapes = stream["cond_image_shapes"]
        roles = stream["cond_image_roles"]
        # Native Ref2VA keyframes precede reference streams and are not part
        # of cond_image_shapes. They stay dense, like FL2VA conditioning rows.
        cursor = conditions.numel() - sum(math.prod(shape) for shape in shapes)
        if cursor < 0 or len(shapes) != len(roles) or num_steps < 1:
            raise ValueError("Invalid H3 stream layout for Spark joint-video attention")
        videos = []
        for shape, role in zip(shapes, roles):
            count = math.prod(shape)
            if role == "independent_cube" and shape[0] > 1:
                videos.append(conditions[cursor : cursor + count])
            cursor += count
        videos.append(img_pos[update])
        sparse = torch.cat(videos)
        used = int(packed["cu_seqlens"][1])
        seq_len = int(packed["seq_len"])
        if (
            not sparse.numel()
            or sparse.min() < 0
            or sparse.max() >= used
            or sparse.unique().numel() != sparse.numel()
        ):
            raise ValueError("Invalid H3 video rows for Spark joint-video attention")
        selected = torch.zeros(used, dtype=torch.bool)
        selected[sparse] = True
        permutation = torch.cat((sparse, torch.arange(used)[~selected]))
        positions = packed["img_position_ids"].cpu().index_select(0, sparse)
        grid = tuple(int(positions[:, axis].unique().numel()) for axis in range(3))
        if math.prod(grid) != sparse.numel():
            # Different reference/target resolutions form a composite grid.
            grid = (1, 1, sparse.numel())
        return SparkAttentionMetadata(
            current_timestep=0,
            num_steps=num_steps,
            used=used,
            seq_len=seq_len,
            permutation=permutation.to(device),
            inverse_permutation=permutation.argsort().to(device),
            video_positions=positions.to(device),
            video_tokens=sparse.numel(),
            grid=grid,
        )


class SparkAttentionBackend(AttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SPARK_ATTN

    @staticmethod
    def get_impl_cls() -> type[SparkAttentionImpl]:
        return SparkAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[SparkAttentionMetadata]:
        return SparkAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[SparkAttentionMetadataBuilder]:
        return SparkAttentionMetadataBuilder


class SparkAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if head_size != 128 or causal or num_kv_heads not in (None, num_heads):
            raise ValueError(
                "Spark-H3 requires noncausal attention with equal Q/KV heads and head_dim=128"
            )
        if not math.isclose(softmax_scale, head_size**-0.5):
            raise ValueError("Spark-H3 requires the default attention scale")
        config = get_global_server_args().attention_backend_config or {}
        self.topk_ratio = float(config.get("topk_ratio", 0.2))
        self.warmup_ratio = float(config.get("warmup_ratio", 0.2))
        self.dense_layers = int(config.get("dense_layers", 0))
        self.min_tokens = int(config.get("min_tokens", 12288))
        if not 0 < self.topk_ratio <= 1 or not 0 <= self.warmup_ratio <= 1:
            raise ValueError(
                "Spark topk_ratio must be in (0, 1] and warmup_ratio in [0, 1]"
            )
        if self.dense_layers < 0 or self.min_tokens < 0:
            raise ValueError("Spark dense_layers and min_tokens must be non-negative")
        match = re.match(r"^blocks\.(\d+)\.", prefix)
        self.layer_idx = int(match[1]) if match else None
        self.dense_impl = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=False,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.dense",
            **extra_impl_args,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if self.layer_idx is not None:
            raise ValueError(
                "Spark-H3 requires packed varlen attention and H3 stream metadata"
            )
        return self.dense_impl.forward(query, key, value, attn_metadata)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if self.layer_idx is not None:
            metadata = get_forward_context().attn_metadata
            if not isinstance(metadata, SparkAttentionMetadata):
                raise ValueError("Spark-H3 requires joint-video request metadata")
            bounds = (0, metadata.used, metadata.seq_len)
            if cu_seqlens_host != bounds or query.shape[0] != metadata.seq_len:
                raise ValueError("Spark-H3 metadata does not match the packed sequence")
            step = get_forward_context().current_timestep
            sparse = (
                step >= math.ceil(metadata.num_steps * self.warmup_ratio)
                and self.layer_idx >= self.dense_layers
                and metadata.used >= self.min_tokens
                and metadata.video_tokens >= 64
            )
            if sparse:
                return self._sparse_attention(query, key, value, metadata, step)
        return self.dense_impl.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )

    def _sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: SparkAttentionMetadata,
        step: int,
    ) -> torch.Tensor:
        from h3_sparse_attention.processor import (
            H3SparseAttentionConfig,
            PackedLayout,
            _Controller,
        )
        from h3_sparse_attention.spark_integration import spark_attention_bthd

        if metadata.controller is None:
            metadata.controller = _Controller(
                H3SparseAttentionConfig.spark(
                    num_denoise_steps=metadata.num_steps,
                    sol_route_topk_ratio=self.topk_ratio,
                    landmark_tree_v2_layout_reuse="q_from_k",
                )
            )
        controller = metadata.controller
        controller.evaluation_index = step
        layout = PackedLayout(
            permutation=metadata.permutation,
            inverse_permutation=metadata.inverse_permutation,
            grid=metadata.grid,
            video_tokens=metadata.video_tokens,
            sequence_length=metadata.used,
            video_positions=metadata.video_positions,
        )
        q, k, v = (
            tensor.index_select(0, metadata.permutation).unsqueeze(0)
            for tensor in (query, key, value)
        )
        result = spark_attention_bthd(controller, q, k, v, layout, self.layer_idx)
        out = torch.empty_like(query)
        out[: metadata.used] = result[0].index_select(0, metadata.inverse_permutation)
        # Padding is a separate attention document, never a Spark sink.
        if metadata.used < metadata.seq_len:
            padding = slice(metadata.used, metadata.seq_len)
            out[padding] = self.dense_impl.forward(
                query[padding][None], key[padding][None], value[padding][None], None
            )[0]
        logger.info_once(
            f"Spark joint-video attention active: video_tokens={metadata.video_tokens}, "
            f"exact_context_tokens={metadata.used - metadata.video_tokens}, "
            f"topk_ratio={self.topk_ratio}, backend={controller.sol_backend}"
        )
        return out
