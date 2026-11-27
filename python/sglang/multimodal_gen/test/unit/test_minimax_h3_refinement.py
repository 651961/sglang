# SPDX-License-Identifier: Apache-2.0
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.refinement import (
    refinement_frames,
    refinement_sigmas,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_spatial_shape,
)


def test_refinement_keeps_trained_canvas_budget_separate_from_base():
    dimensions = {"width": 16, "height": 9, "base_short_edge": 1448}
    base = minimax_h3_resolve_spatial_shape(**dimensions)
    refined = minimax_h3_resolve_spatial_shape(**dimensions, refinement=True)
    assert (base["width"], base["height"]) == (1344, 768)
    assert (refined["width"], refined["height"]) == (1920, 1056)


def test_refinement_couples_video_to_training_audio_timestep():
    sigmas = refinement_sigmas(8)
    assert sigmas["video"] == pytest.approx(np.linspace(0.5, 0.0, 9))
    # Independent evaluation of VeOmni's descending, shifted training table.
    base = torch.linspace(1, 0, 1001)[:-1]
    table = (3000 * base / (1 + 2 * base)).bfloat16()
    idx = [round(sigma * 999) for sigma in sigmas["video"]]
    assert sigmas["audio"] == pytest.approx((table[idx].float() / 1000).tolist())


def test_refinement_replaces_only_first_resized_frame(tmp_path):
    source = tmp_path / "coarse.mkv"
    frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
    frames[..., 2] = 255
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "16x16",
            "-r",
            "24",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            str(source),
        ],
        input=frames.tobytes(),
        check=True,
    )
    anchor = Image.new("RGB", (64, 32), "red")
    result = refinement_frames(str(source), anchor, 5)
    assert result.shape == (5, 32, 64, 3)
    np.testing.assert_array_equal(result[0], np.asarray(anchor))
    assert (result[1:, :, :, 2] == 255).all()


def test_peer_partial_fast_path_skips_lora_wrapped_output_projection():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTBlock

    # LoRA wrappers intentionally expose the TP group only on their base layer.
    block = SimpleNamespace(
        attn=SimpleNamespace(hybrid=None, out_proj=torch.nn.Identity()),
        mlp=SimpleNamespace(fc2=torch.nn.Identity()),
    )
    activation = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    assert not MiniMaxH3DiTBlock._takes_peer_partials(block, activation)
