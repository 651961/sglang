# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the native SenseNova-U1.5 rollout bridge.

The native pipeline normally imports the complete SGLang runtime, including optional
GPU kernels.  These tests load the rollout modules with small boundary stubs so the
real controller and scheduler math can be exercised without importing ``sgl_kernel``
or constructing a model.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image


REPO_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "python" / "sglang").is_dir()
    ),
    Path.cwd(),
)
PYTHON_ROOT = REPO_ROOT / "python"


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _leaf(name: str, **members) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(members)
    return module


def _load_source(name: str, relative_path: str) -> types.ModuleType:
    path = PYTHON_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
) -> tuple[int, int]:
    """The arithmetic contract used by qwen_vl.smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = math.floor(height / beta / factor) * factor
        resized_w = math.floor(width / beta / factor) * factor
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return resized_h, resized_w


class _OutputBatch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _PipelineStage:
    def __init__(self):
        self.server_args = None

    @staticmethod
    def _component_stage_name(stage_name):
        return stage_name


class _IdentityPipelineConfig:
    @staticmethod
    def shard_latents_for_sp(*, batch, latents):
        return latents, None

    @staticmethod
    def gather_latents_for_sp(latents, *, batch):
        return latents


@pytest.fixture(scope="module")
def u1_module():
    """Load the real rollout code while replacing heavyweight runtime boundaries."""
    package_names = [
        "sglang",
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.managers",
        "sglang.multimodal_gen.runtime.managers.memory_managers",
        "sglang.multimodal_gen.runtime.pipelines_core",
        "sglang.multimodal_gen.runtime.pipelines_core.stages",
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages",
        "sglang.multimodal_gen.runtime.post_training",
        "sglang.srt",
        "sglang.srt.multimodal",
        "sglang.srt.multimodal.processors",
    ]
    modules = {name: _package(name) for name in package_names}

    @contextlib.contextmanager
    def forward_context(**_kwargs):
        yield

    class ComponentUse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    modules.update(
        {
            "sglang.multimodal_gen.runtime.distributed": _leaf(
                "sglang.multimodal_gen.runtime.distributed",
                get_local_torch_device=lambda: torch.device("cpu"),
                get_sp_world_size=lambda: 1,
            ),
            "sglang.multimodal_gen.runtime.managers.forward_context": _leaf(
                "sglang.multimodal_gen.runtime.managers.forward_context",
                set_forward_context=forward_context,
            ),
            "sglang.multimodal_gen.runtime.managers.memory_managers.component_manager": _leaf(
                "sglang.multimodal_gen.runtime.managers.memory_managers.component_manager",
                ComponentUse=ComponentUse,
            ),
            "sglang.multimodal_gen.runtime.pipelines_core.schedule_batch": _leaf(
                "sglang.multimodal_gen.runtime.pipelines_core.schedule_batch",
                OutputBatch=_OutputBatch,
                Req=SimpleNamespace,
            ),
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base": _leaf(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.base",
                PipelineStage=_PipelineStage,
            ),
            "sglang.multimodal_gen.runtime.post_training.sp_utils": _leaf(
                "sglang.multimodal_gen.runtime.post_training.sp_utils",
                gather_stacked_latents_for_sp=lambda *, stacked_latents, **_kwargs: stacked_latents,
            ),
            "sglang.multimodal_gen.runtime.server_args": _leaf(
                "sglang.multimodal_gen.runtime.server_args",
                ServerArgs=SimpleNamespace,
            ),
            "sglang.srt.multimodal.processors.qwen_vl": _leaf(
                "sglang.srt.multimodal.processors.qwen_vl",
                smart_resize=_smart_resize,
            ),
        }
    )

    with patch.dict(sys.modules, modules):
        _load_source(
            "sglang.multimodal_gen.runtime.post_training.rl_dataclasses",
            "sglang/multimodal_gen/runtime/post_training/rl_dataclasses.py",
        )
        _load_source(
            "sglang.multimodal_gen.runtime.post_training.scheduler_rl_debug_mixin",
            "sglang/multimodal_gen/runtime/post_training/scheduler_rl_debug_mixin.py",
        )
        _load_source(
            "sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin",
            "sglang/multimodal_gen/runtime/post_training/scheduler_rl_mixin.py",
        )
        _load_source(
            "sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin",
            "sglang/multimodal_gen/runtime/post_training/rollout_denoising_mixin.py",
        )
        target = _load_source(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_5",
            "sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sensenova_u1_5.py",
        )
        yield target


def _batch(**overrides) -> SimpleNamespace:
    values = dict(
        rollout=True,
        rollout_sde_type="ode",
        rollout_sde_step_indices=[1],
        rollout_noise_level=0.7,
        rollout_log_prob_no_const=True,
        rollout_debug_mode=False,
        rollout_return_denoising_env=False,
        rollout_return_dit_trajectory=False,
        rollout_return_transition_pairs=True,
        rollout_return_step_indices=None,
        rollout_trajectory_data=None,
        target_pixels=2048 * 2048,
        input_max_pixels="auto",
        timestep_shift=3.0,
        t_eps=0.02,
        cfg_norm="none",
        cfg_scale=4.0,
        img_cfg_scale=1.0,
        scheduler=None,
        latents=None,
        did_sp_shard_latents=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stage_and_server(u1_module):
    model = SimpleNamespace(patch_size=16, downsample_ratio=0.5)
    stage = u1_module.SenseNovaU1_5GenerationStage(model, tokenizer=None)
    server = SimpleNamespace(pipeline_config=_IdentityPipelineConfig())
    stage.server_args = server
    return stage, server


def _controller(u1_module, batch, *, batch_size=1):
    stage, server = _stage_and_server(u1_module)
    generators = [torch.Generator().manual_seed(100 + i) for i in range(batch_size)]
    controller = u1_module._SenseNovaU1_5RolloutController(
        stage,
        batch,
        server,
        generators,
        prompt="make it red",
        image_paths=["input.png"],
        preprocessed_image_sizes=[(640, 384)],
        output_size=(640, 384),
    )
    return controller


def test_sigma_complement_and_negative_velocity_mapping(u1_module):
    batch = _batch(rollout_sde_type="cps", rollout_sde_step_indices=[1])
    controller = _controller(u1_module, batch)
    timesteps = torch.tensor([0.0, 0.2, 0.55, 1.0])
    image = torch.randn(1, 3, 2, 2)
    velocity = torch.randn_like(image)
    controller.start(image, noise_scale=8.0, timesteps=timesteps)

    captured = {}

    def sample(batch_arg, **kwargs):
        captured["batch"] = batch_arg
        captured.update(kwargs)
        return kwargs["sample"] + 1.0

    controller.scheduler.flow_sde_sampling = sample
    native_next = image + (timesteps[2] - timesteps[1]) * velocity
    result = controller.step(
        1, image, velocity, timesteps[1], timesteps[2], native_next
    )

    torch.testing.assert_close(controller.sigmas, 1.0 - timesteps, rtol=0, atol=0)
    assert captured["batch"] is batch
    torch.testing.assert_close(captured["model_output"], -velocity, rtol=0, atol=0)
    assert captured["current_sigma"].item() == pytest.approx(0.8)
    assert captured["next_sigma"].item() == pytest.approx(0.45)
    assert captured["base_noise_scale"] == pytest.approx(8.0)
    torch.testing.assert_close(result, image + 1.0)


def test_ode_controller_preserves_original_u1_euler_step(u1_module):
    batch = _batch(rollout_sde_type="ode", rollout_sde_step_indices=[2])
    controller = _controller(u1_module, batch)
    timesteps = torch.tensor([0.0, 0.1, 0.4, 0.9, 1.0])
    image = torch.randn(1, 3, 3, 2, dtype=torch.float32)
    velocity = torch.randn_like(image)
    controller.start(image, noise_scale=8.0, timesteps=timesteps)

    expected = image + (timesteps[3] - timesteps[2]) * velocity
    actual = controller.step(
        2, image, velocity, timesteps[2], timesteps[3], expected
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sde_type", ["sde", "cps"])
def test_stochastic_increment_uses_resolution_noise_scale(u1_module, sde_type):
    scheduler = u1_module._SenseNovaU1_5RolloutScheduler(
        torch.tensor([1.0, 0.8, 0.5, 0.0])
    )
    shape = (1, 3, 2, 2)
    batch = _batch(
        rollout_sde_type=sde_type,
        rollout_sde_step_indices=[1],
        rollout_log_prob_no_const=False,
        rollout_debug_mode=True,
        latents=torch.zeros(shape),
    )
    scheduler.prepare_rollout(batch, pipeline_config=_IdentityPipelineConfig())
    batch._rollout_loop_step_index = 1
    variance_noise = torch.ones(shape)

    def fixed_noise(batch_arg, *_args, **_kwargs):
        scheduler._get_rollout_session_data(batch_arg).noise_buffer = variance_noise
        return variance_noise

    scheduler._rollout_variance_noise = fixed_noise
    sample = torch.randn(shape)
    model_output = torch.randn(shape)
    result = scheduler.flow_sde_sampling(
        batch,
        model_output=model_output,
        sample=sample,
        current_sigma=torch.tensor(0.8),
        next_sigma=torch.tensor(0.5),
        generator=torch.Generator().manual_seed(0),
        base_noise_scale=8.0,
    )
    _, means, noise_std_devs, _ = scheduler.consume_local_rollout_debug_tensors(
        batch
    )

    torch.testing.assert_close(
        result - means[:, 0],
        torch.ones_like(result) * noise_std_devs[:, 0].reshape(1, 1, 1, 1),
    )
    scheduler.release_rollout_resources(batch)


def test_controller_rejects_ambiguous_trajectory_filters(u1_module):
    image = torch.zeros(1, 3, 2, 2)
    timesteps = torch.linspace(0, 1, 4)
    full_trajectory = _controller(
        u1_module,
        _batch(rollout_return_dit_trajectory=True),
    )
    with pytest.raises(ValueError, match="full RGB DiT trajectory"):
        full_trajectory.start(image, noise_scale=8.0, timesteps=timesteps)

    return_filter = _controller(
        u1_module,
        _batch(rollout_return_step_indices=[1]),
    )
    with pytest.raises(ValueError, match="rollout_return_step_indices"):
        return_filter.start(image, noise_scale=8.0, timesteps=timesteps)


def test_ode_requires_no_constant_log_prob_mode(u1_module):
    controller = _controller(
        u1_module,
        _batch(
            rollout_sde_type="ode",
            rollout_log_prob_no_const=False,
        ),
    )
    with pytest.raises(ValueError, match="rollout_log_prob_no_const=True"):
        controller.start(
            torch.zeros(1, 3, 2, 2),
            noise_scale=8.0,
            timesteps=torch.linspace(0, 1, 4),
        )


def test_cps_rejects_terminal_transition(u1_module):
    batch = _batch(rollout_sde_type="cps", rollout_sde_step_indices=[3])
    controller = _controller(u1_module, batch)
    with pytest.raises(ValueError, match="next_sigma=0"):
        controller.start(
            torch.zeros(1, 3, 2, 2),
            noise_scale=8.0,
            timesteps=torch.linspace(0, 1, 5),
        )


def test_selected_transitions_are_explicit_pairs(u1_module):
    batch = _batch(
        rollout_sde_type="ode",
        rollout_sde_step_indices=[3, 1, 3],
    )
    controller = _controller(u1_module, batch)
    timesteps = torch.tensor([0.0, 0.1, 0.35, 0.7, 1.0])
    image = torch.zeros(1, 3, 2, 2)
    velocity = torch.ones_like(image)
    controller.start(image, noise_scale=8.0, timesteps=timesteps)

    states = [image]
    for step_index in range(4):
        states.append(
            controller.step(
                step_index,
                states[-1],
                velocity,
                timesteps[step_index],
                timesteps[step_index + 1],
                states[-1]
                + (timesteps[step_index + 1] - timesteps[step_index]) * velocity,
            )
        )
    controller.finalize()

    pairs = batch.rollout_trajectory_data.transition_pairs
    assert pairs.step_indices.tolist() == [1, 3]
    assert pairs.latents.shape == (1, 2, 3, 2, 2)
    assert pairs.next_latents.shape == (1, 2, 3, 2, 2)
    torch.testing.assert_close(pairs.latents[0, 0], states[1][0])
    torch.testing.assert_close(pairs.next_latents[0, 0], states[2][0])
    torch.testing.assert_close(pairs.latents[0, 1], states[3][0])
    torch.testing.assert_close(pairs.next_latents[0, 1], states[4][0])
    sigmas = 1.0 - timesteps
    torch.testing.assert_close(pairs.timesteps, timesteps[[1, 3]])
    torch.testing.assert_close(pairs.next_timesteps, timesteps[[2, 4]])
    torch.testing.assert_close(pairs.sigmas, sigmas[[1, 3]])
    torch.testing.assert_close(pairs.next_sigmas, sigmas[[2, 4]])
    assert pairs.base_noise_scale == pytest.approx(8.0)


def test_batched_stochastic_action_is_saved_before_model_dtype_cast(u1_module):
    batch = _batch(
        rollout_sde_type="sde",
        rollout_sde_step_indices=[1],
        rollout_return_transition_pairs=True,
    )
    controller = _controller(u1_module, batch, batch_size=2)
    timesteps = torch.tensor([0.0, 0.3, 0.7, 1.0])
    image = torch.zeros(2, 3, 2, 2, dtype=torch.bfloat16)
    velocity = torch.ones_like(image)
    controller.start(image, noise_scale=8.0, timesteps=timesteps)

    def fp32_action(batch_arg, *, sample, **_kwargs):
        batch_size = sample.shape[0]
        controller.scheduler.append_local_rollout_log_probs(
            batch_arg,
            torch.zeros(batch_size, dtype=torch.float32),
            torch.full((batch_size,), float(sample[0].numel())),
        )
        return sample.float() + 0.125

    controller.scheduler.flow_sde_sampling = fp32_action
    state = image
    for step_index in range(3):
        native_next = state + (
            timesteps[step_index + 1] - timesteps[step_index]
        ) * velocity
        state = controller.step(
            step_index,
            state,
            velocity,
            timesteps[step_index],
            timesteps[step_index + 1],
            native_next,
        )
    controller.finalize()

    pairs = batch.rollout_trajectory_data.transition_pairs
    assert pairs.latents.shape == (2, 1, 3, 2, 2)
    assert pairs.next_latents.shape == (2, 1, 3, 2, 2)
    assert pairs.latents.dtype == torch.bfloat16
    assert pairs.next_latents.dtype == torch.float32


def test_batched_generators_are_independent_and_reproducible(u1_module):
    scheduler_cls = u1_module._SenseNovaU1_5RolloutScheduler
    shape = (2, 3, 4, 5)

    def draw():
        scheduler = scheduler_cls(torch.tensor([1.0, 0.5, 0.0]))
        batch = _batch(latents=torch.zeros(shape))
        scheduler.prepare_rollout(batch, pipeline_config=_IdentityPipelineConfig())
        generators = [
            torch.Generator().manual_seed(17),
            torch.Generator().manual_seed(29),
        ]
        return scheduler._rollout_variance_noise(
            batch, torch.zeros(shape), generators
        ).clone()

    first = draw()
    second = draw()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first[0], first[1])
    expected_0 = torch.randn((1, *shape[1:]), generator=torch.Generator().manual_seed(17))
    expected_1 = torch.randn((1, *shape[1:]), generator=torch.Generator().manual_seed(29))
    torch.testing.assert_close(first[0:1], expected_0, rtol=0, atol=0)
    torch.testing.assert_close(first[1:2], expected_1, rtol=0, atol=0)


def test_group_guidance_has_an_explicit_batch_dimension(u1_module):
    batch = _batch(
        rollout_return_denoising_env=True,
        rollout_return_transition_pairs=False,
    )
    controller = _controller(u1_module, batch, batch_size=2)
    controller.start(
        torch.zeros(2, 3, 2, 2),
        noise_scale=8.0,
        timesteps=torch.linspace(0, 1, 4),
    )
    guidance = batch._rollout_denoising_env_state["env"].guidance
    image_kwargs = batch._rollout_denoising_env_state["env"].image_kwargs
    assert guidance.shape == (2, 2)
    torch.testing.assert_close(
        guidance,
        torch.tensor([[4.0, 1.0], [4.0, 1.0]]),
    )
    assert image_kwargs["stage_preprocessed_image_sizes"] == [(640, 384)]
    assert image_kwargs["model_input_size"] == image_kwargs["output_size"]
    assert image_kwargs["rollout_sde_type"] == "ode"
    assert image_kwargs["rollout_noise_level"] == 0.7
    assert image_kwargs["rollout_log_prob_no_const"] is True


@pytest.mark.parametrize("original_size", [(1920, 1080), (1537, 977), (2048, 2048)])
def test_dynamic_edit_sizes_remain_multiples_of_32(u1_module, original_size):
    image = Image.new("RGB", original_size)
    resized = u1_module._resize_input_images([image], 2048 * 2048)[0]
    batch = SimpleNamespace(
        width=2048,
        height=2048,
        target_pixels=2048 * 2048,
        sampling_params=SimpleNamespace(_explicit_fields=set()),
    )
    output_width, output_height = u1_module.SenseNovaU1_5GenerationStage._resolve_size(
        batch, [resized]
    )
    model_input_height, model_input_width = _smart_resize(
        resized.height,
        resized.width,
        factor=32,
        min_pixels=512 * 512,
        max_pixels=2048 * 2048,
    )

    assert resized.width % 32 == resized.height % 32 == 0
    assert (output_width, output_height) == (
        model_input_width,
        model_input_height,
    )
    assert output_width % 32 == output_height % 32 == 0
    assert output_width / output_height == pytest.approx(
        original_size[0] / original_size[1], rel=0.04
    )


def test_non_rollout_forward_does_not_create_trajectory(u1_module):
    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.kwargs = None

        def t2i_generate(self, _tokenizer, _prompt, **kwargs):
            self.kwargs = kwargs
            height, width = kwargs["image_size"][1], kwargs["image_size"][0]
            return torch.zeros(kwargs["batch_size"], 3, height, width)

    model = FakeModel()
    stage = u1_module.SenseNovaU1_5GenerationStage(model, tokenizer=None)
    batch = SimpleNamespace(
        prompt="a red cube",
        image_path=None,
        input_max_pixels="auto",
        width=32,
        height=32,
        sampling_params=SimpleNamespace(_explicit_fields={"width", "height"}),
        rollout=False,
        cfg_scale=4.0,
        img_cfg_scale=1.0,
        cfg_norm="none",
        timestep_shift=3.0,
        t_eps=0.02,
        num_inference_steps=2,
        num_outputs_per_prompt=1,
        seed=7,
        trajectory_timesteps=None,
        trajectory_latents=None,
        rollout_trajectory_data=None,
        metrics=None,
    )
    result = stage.forward(batch, SimpleNamespace())

    assert result.rollout_trajectory_data is None
    assert model.kwargs["generators"] is None
    assert model.kwargs["denoise_start_callback"] is None
    assert model.kwargs["denoise_step_callback"] is None
