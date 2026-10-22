# SPDX-License-Identifier: Apache-2.0
"""Spark joint-video roles, request scheduling, and packed-document isolation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.spark_attn import (
    SparkAttentionImpl,
    SparkAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _build_h3_attn_metadata,
)


def _packed(*, refs=False, compressed=False, keyframes=False, reference_size=16):
    common = dict(text_len=3, latent_t=5, latent_h=16, latent_w=16, audio_t=5)
    if not refs:
        return minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=keyframes,
            frame_count=22,
            keyframe_frame_indices=[0, -1] if keyframes else None,
        )
    video = dict(
        kind="video_audio",
        ref_audio_t=3,
        latent_t=3,
        latent_h=reference_size,
        latent_w=reference_size,
    )
    if compressed:
        video.update(dense_latent_t=5, dense_indices=[0, 1, 3])
    return minimax_h3_packed_sequence_ref2va_blocks(
        **common,
        keyframe_frame_indices=[0, -1] if keyframes else None,
        frame_count=22,
        ref_blocks=[
            dict(kind="image", latent_h=8, latent_w=8),
            dict(kind="audio", ref_audio_t=2),
            video,
            dict(kind="video", ref_audio_t=0, latent_t=1, latent_h=8, latent_w=8),
        ],
    )


def _metadata(packed, device="cpu"):
    return SparkAttentionMetadataBuilder().build(
        packed=packed,
        num_steps=19,
        device=torch.device(device),
    )


def _impl(monkeypatch, *, prefix="blocks.0.attn", **config):
    module = "sglang.multimodal_gen.runtime.layers.attention.backends.spark_attn"
    monkeypatch.setattr(
        f"{module}.get_global_server_args",
        lambda: SimpleNamespace(attention_backend_config=config),
    )
    return SparkAttentionImpl(2, 128, 128**-0.5, prefix=prefix)


@pytest.mark.parametrize("refs", [False, True])
@pytest.mark.parametrize("keyframes", [False, True])
def test_joint_video_roles_preserve_original_coordinates(refs, keyframes):
    packed = _packed(refs=refs, compressed=refs, keyframes=keyframes)
    metadata = _metadata(packed)
    target = packed["img_pos"][packed["update_mask"]]
    # Only the multi-frame reference video joins the target. Images, one-time
    # references, and FL2VA/Ref2VA keyframes stay outside the sparse domain.
    if refs:
        reference_start = 3 + (128 if keyframes else 0) + 16 + 4 + 6
        reference = torch.arange(reference_start, reference_start + 192)
        expected = torch.cat((reference, target))
    else:
        expected = target
    assert torch.equal(metadata.permutation[: metadata.video_tokens], expected)
    assert metadata.video_tokens == (512 if refs else 320)
    torch.testing.assert_close(
        metadata.video_positions,
        packed["img_position_ids"][expected],
        rtol=0,
        atol=0,
    )
    assert torch.equal(metadata.permutation.sort().values, torch.arange(metadata.used))
    assert torch.equal(
        metadata.permutation[metadata.inverse_permutation],
        torch.arange(metadata.used),
    )
    assert metadata.controller is None


def test_joint_video_supports_different_reference_resolution():
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=5,
        latent_h=16,
        latent_w=16,
        audio_t=5,
        ref_blocks=[
            dict(kind="video", ref_audio_t=0, latent_t=3, latent_h=8, latent_w=8)
        ],
    )
    metadata = _metadata(packed)
    assert metadata.video_tokens == 368
    assert metadata.grid == (1, 1, 368)


def test_h3_backend_override_builds_spark_metadata():
    server_args = SimpleNamespace(
        attention_backend="fa",
        component_attention_backends={"transformer": "spark_attn"},
    )
    packed = _packed(refs=True)
    metadata = _build_h3_attn_metadata(
        server_args,
        packed=packed,
        num_steps=19,
        device=torch.device("cpu"),
    )
    assert metadata.video_tokens == 512
    assert metadata.num_steps == 19


def test_dense_warmup_uses_actual_request_steps(monkeypatch):
    impl = _impl(monkeypatch, min_tokens=0)
    packed = _packed(refs=True)
    metadata = _metadata(packed)
    query = torch.randn(metadata.seq_len, 2, 128)
    sparse = Mock(return_value=torch.zeros_like(query))
    monkeypatch.setattr(impl, "_sparse_attention", sparse)
    kwargs = dict(
        cu_seqlens=packed["cu_seqlens"],
        max_seqlen=metadata.used,
        cu_seqlens_host=(0, metadata.used, metadata.seq_len),
    )
    for step in range(4):
        with set_forward_context(step, metadata):
            result = impl.forward_varlen(query, query, query, **kwargs)
        assert result.abs().sum() > 0
    sparse.assert_not_called()
    with set_forward_context(4, metadata):
        impl.forward_varlen(query, query, query, **kwargs)
    assert sparse.call_count == 1
    # A fresh short request recomputes ceil(3*0.2)=1 and owns fresh state.
    metadata = _metadata(packed)
    metadata.num_steps = 3
    with set_forward_context(0, metadata):
        impl.forward_varlen(query, query, query, **kwargs)
    assert sparse.call_count == 1
    with set_forward_context(1, metadata):
        impl.forward_varlen(query, query, query, **kwargs)
    assert sparse.call_count == 2


def test_token_refiner_stays_dense_without_forward_context(monkeypatch):
    impl = _impl(monkeypatch, prefix="token_refiner.blocks.0.attn")
    query = torch.randn(3, 2, 128)
    out = impl.forward_varlen(
        query,
        query,
        query,
        cu_seqlens=torch.tensor([0, 3, 3], dtype=torch.int32),
        cu_seqlens_host=(0, 3, 3),
        max_seqlen=3,
    )
    assert out.shape == query.shape


@pytest.mark.parametrize(
    "config", [{"min_tokens": 12288}, {"dense_layers": 1, "min_tokens": 0}]
)
def test_short_sequences_and_leading_dense_layers_keep_dense(monkeypatch, config):
    impl = _impl(monkeypatch, warmup_ratio=0, **config)
    packed = _packed(refs=True)
    metadata = _metadata(packed)
    query = torch.randn(metadata.seq_len, 2, 128)
    sparse = Mock(side_effect=AssertionError("Sparse attention must not run"))
    monkeypatch.setattr(impl, "_sparse_attention", sparse)
    with set_forward_context(0, metadata):
        impl.forward_varlen(
            query,
            query,
            query,
            cu_seqlens=packed["cu_seqlens"],
            max_seqlen=metadata.used,
            cu_seqlens_host=(0, metadata.used, metadata.seq_len),
        )
    sparse.assert_not_called()


def test_missing_or_mismatched_metadata_is_rejected(monkeypatch):
    impl = _impl(monkeypatch, warmup_ratio=0, min_tokens=0)
    packed = _packed(refs=True)
    metadata = _metadata(packed)
    query = torch.randn(metadata.seq_len, 2, 128)
    kwargs = dict(
        cu_seqlens=packed["cu_seqlens"],
        max_seqlen=metadata.used,
        cu_seqlens_host=(0, metadata.used, metadata.seq_len),
    )
    with (
        set_forward_context(0, None),
        pytest.raises(ValueError, match="request metadata"),
    ):
        impl.forward_varlen(query, query, query, **kwargs)
    kwargs["cu_seqlens_host"] = (0, metadata.seq_len)
    with (
        set_forward_context(0, metadata),
        pytest.raises(ValueError, match="packed sequence"),
    ):
        impl.forward_varlen(query, query, query, **kwargs)


@pytest.mark.parametrize(
    "config",
    [
        {"topk_ratio": 0},
        {"topk_ratio": 1.1},
        {"warmup_ratio": -0.1},
        {"warmup_ratio": 1.1},
        {"dense_layers": -1},
        {"min_tokens": -1},
    ],
)
def test_invalid_spark_config_is_rejected(monkeypatch, config):
    with pytest.raises(ValueError):
        _impl(monkeypatch, **config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("topk_ratio", [1.0, 0.2])
@pytest.mark.parametrize("reference_size", [8, 16])
def test_spark_cuda_preserves_dense_conditions_and_padding(
    monkeypatch, topk_ratio, reference_size
):
    pytest.importorskip("h3_sparse_attention")
    if torch.cuda.get_device_capability() not in {(8, 0), (8, 6), (8, 9), (12, 0)}:
        pytest.skip("Spark production kernel architecture required")
    # Small test grids normally select upstream's streamed compatibility path.
    # Exercise the same fused Reweight kernel used by production video lengths.
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    impl = _impl(monkeypatch, topk_ratio=topk_ratio, warmup_ratio=0, min_tokens=0)
    packed = _packed(
        refs=True, compressed=True, keyframes=True, reference_size=reference_size
    )
    metadata = _metadata(packed, "cuda")
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(metadata.seq_len, 2, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    kwargs = dict(
        cu_seqlens=packed["cu_seqlens"].cuda(),
        max_seqlen=metadata.used,
        cu_seqlens_host=(0, metadata.used, metadata.seq_len),
    )
    expected = impl.dense_impl.forward_varlen(q, k, v, **kwargs)
    originals = [tensor.clone() for tensor in (q, k, v)]
    with torch.inference_mode(), set_forward_context(0, metadata):
        result = impl.forward_varlen(q, k, v, **kwargs)
    assert result.isfinite().all()
    assert metadata.controller.counts["sol_landmark_preprocess_calls"] == 1
    assert metadata.controller.counts["sol_virtual_query_calls"] == 1
    assert metadata.controller.sol_backend.endswith("fused_virtual_query")
    dense_rows = metadata.permutation[metadata.video_tokens :]
    torch.testing.assert_close(
        result[dense_rows], expected[dense_rows], atol=0.004, rtol=0.03
    )
    torch.testing.assert_close(result[metadata.used :], expected[metadata.used :])
    if topk_ratio == 1:
        torch.testing.assert_close(result, expected, atol=0.004, rtol=0.03)
    for tensor, original in zip((q, k, v), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
