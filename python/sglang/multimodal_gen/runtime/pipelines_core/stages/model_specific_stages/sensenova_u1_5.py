# SPDX-License-Identifier: Apache-2.0
"""Single-stage non-thinking SenseNova-U1.5 generation/editing stage."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutTrajectoryData,
    RolloutTransitionPairs,
)
from sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin import (
    RolloutDenoisingMixin,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.multimodal.processors.qwen_vl import smart_resize

_DEFAULT_INPUT_PIXELS = 2048 * 2048
_MIN_INPUT_PIXELS = 512 * 512


class _SenseNovaU1_5RolloutScheduler(SchedulerRLMixin):
    """Minimal scheduler state for U1.5's model-owned denoising loop."""

    def __init__(self, sigmas: torch.Tensor):
        self.sigmas = sigmas


class _SenseNovaU1_5RolloutController:
    """Bridge U1.5's pixel-space denoising loop to SGLang rollout hooks."""

    def __init__(
        self,
        stage: "SenseNovaU1_5GenerationStage",
        batch: Req,
        server_args: ServerArgs,
        generators: list[torch.Generator],
        *,
        prompt: str,
        image_paths: list[str],
        preprocessed_image_sizes: list[tuple[int, int]],
        output_size: tuple[int, int],
    ) -> None:
        self.stage = stage
        self.batch = batch
        self.server_args = server_args
        self.generators = generators
        self.prompt = prompt
        self.image_paths = image_paths
        self.preprocessed_image_sizes = preprocessed_image_sizes
        self.output_size = output_size
        self.scheduler: _SenseNovaU1_5RolloutScheduler | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.noise_scale = 1.0
        self.selected_step_indices: list[int] = []
        self.selected_latents: list[torch.Tensor] = []
        self.selected_next_latents: list[torch.Tensor] = []

    def start(
        self, image: torch.Tensor, noise_scale: float, timesteps: torch.Tensor
    ) -> None:
        num_steps = int(timesteps.shape[0]) - 1
        selected = self.batch.rollout_sde_step_indices
        if selected is None:
            raise ValueError(
                "SenseNova-U1.5 rollout requires rollout_sde_step_indices; "
                "returning every 2K RGB transition is prohibitively large."
            )
        self.selected_step_indices = sorted({int(index) for index in selected})
        if not self.selected_step_indices:
            raise ValueError("rollout_sde_step_indices must not be empty")
        if self.batch.rollout_return_dit_trajectory:
            raise ValueError(
                "SenseNova-U1.5 does not return a full RGB DiT trajectory; use "
                "rollout_return_transition_pairs for selected transitions"
            )
        if self.batch.rollout_return_step_indices is not None:
            raise ValueError(
                "SenseNova-U1.5 transition pairs are selected by "
                "rollout_sde_step_indices; rollout_return_step_indices is not supported"
            )
        if (
            self.batch.rollout_sde_type == "ode"
            and not self.batch.rollout_log_prob_no_const
        ):
            raise ValueError(
                "ODE log-probability is defined only in no-constant mode; set "
                "rollout_log_prob_no_const=True"
            )
        invalid = [
            index
            for index in self.selected_step_indices
            if index < 0 or index >= num_steps
        ]
        if invalid:
            raise ValueError(
                f"rollout_sde_step_indices contains out-of-range indices {invalid}; "
                f"valid range is [0, {num_steps - 1}]"
            )
        if (
            self.batch.rollout_sde_type == "cps"
            and num_steps - 1 in self.selected_step_indices
        ):
            raise ValueError(
                "CPS cannot train on the terminal U1.5 step because next_sigma=0 "
                "makes the transition deterministic"
            )

        self.timesteps = timesteps.detach().to(dtype=torch.float32)
        self.sigmas = 1.0 - self.timesteps
        self.noise_scale = float(noise_scale)
        self.scheduler = _SenseNovaU1_5RolloutScheduler(self.sigmas)
        self.batch.scheduler = self.scheduler
        self.batch.latents = image
        self.stage._maybe_prepare_rollout(self.batch)

        width, height = self.output_size
        image_kwargs: dict[str, Any] = {
            "model_family": "sensenova_u1_5",
            "image_paths": self.image_paths,
            "stage_preprocessed_image_sizes": self.preprocessed_image_sizes,
            "model_input_size": (width, height),
            "output_size": (width, height),
            "target_pixels": int(
                self.batch.target_pixels
                if self.batch.target_pixels is not None
                else _DEFAULT_INPUT_PIXELS
            ),
            "input_max_pixels": self.batch.input_max_pixels,
            "patch_size": int(self.stage.model.patch_size),
            "downsample_ratio": float(self.stage.model.downsample_ratio),
            "noise_scale": float(noise_scale),
            "timestep_shift": float(self.batch.timestep_shift),
            "t_eps": float(self.batch.t_eps),
            "cfg_norm": str(self.batch.cfg_norm),
            "num_inference_steps": num_steps,
            "selected_step_indices": self.selected_step_indices,
            "rollout_sde_type": str(self.batch.rollout_sde_type),
            "rollout_noise_level": float(self.batch.rollout_noise_level),
            "rollout_log_prob_no_const": bool(
                self.batch.rollout_log_prob_no_const
            ),
        }
        self.stage._maybe_init_denoising_env_collection(
            batch=self.batch,
            pipeline_config=self.server_args.pipeline_config,
            image_kwargs=image_kwargs,
            pos_cond_kwargs={
                "conditioning_mode": "image_and_text",
                "prompt": self.prompt,
            },
            neg_cond_kwargs={
                "conditioning_mode": "image_only",
                "prompt": "",
            },
            guidance=torch.tensor(
                [[float(self.batch.cfg_scale), float(self.batch.img_cfg_scale)]]
                * int(image.shape[0]),
                dtype=torch.float32,
            ),
        )

    def _record_ode_step(
        self,
        image: torch.Tensor,
        velocity: torch.Tensor,
        next_image: torch.Tensor,
    ) -> None:
        assert self.scheduler is not None
        batch_size = int(image.shape[0])
        device = image.device
        self.scheduler.append_local_rollout_log_probs(
            self.batch,
            torch.zeros(batch_size, device=device, dtype=torch.float32),
            torch.full(
                (batch_size,),
                float(math.prod(image.shape[1:])),
                device=device,
                dtype=torch.float32,
            ),
        )
        if (
            self.batch.rollout_debug_mode
            and self.batch._rollout_loop_step_index in self.selected_step_indices
        ):
            self.scheduler.append_local_rollout_debug_tensors(
                self.batch,
                variance_noise=torch.zeros_like(image),
                prev_sample_mean=next_image,
                noise_std_dev=torch.zeros((), device=device, dtype=image.dtype),
                model_output=-velocity,
            )

    def step(
        self,
        step_index: int,
        image: torch.Tensor,
        velocity: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        native_next_image: torch.Tensor,
    ) -> torch.Tensor:
        assert self.scheduler is not None and self.sigmas is not None
        self.batch._rollout_loop_step_index = step_index
        selected = step_index in self.selected_step_indices
        if selected and self.batch.rollout_return_transition_pairs:
            self.selected_latents.append(image.detach().cpu())

        if self.batch.rollout_sde_type == "ode" or not selected:
            next_image_action = native_next_image
            self._record_ode_step(image, velocity, next_image_action)
        else:
            next_image_action = self.scheduler.flow_sde_sampling(
                self.batch,
                model_output=-velocity,
                sample=image,
                current_sigma=self.sigmas[step_index],
                next_sigma=self.sigmas[step_index + 1],
                generator=self.generators,
                base_noise_scale=self.noise_scale,
            )

        if selected and self.batch.rollout_return_transition_pairs:
            self.selected_next_latents.append(next_image_action.detach().cpu())
        return next_image_action.to(dtype=image.dtype)

    def finalize(self) -> None:
        assert (
            self.scheduler is not None
            and self.sigmas is not None
            and self.timesteps is not None
        )
        self.stage._maybe_collect_rollout_log_probs(self.batch)
        if self.batch.rollout_debug_mode:
            assert self.batch.rollout_trajectory_data is not None
            debug_tensors = self.batch.rollout_trajectory_data.rollout_debug_tensors
            assert debug_tensors is not None
            debug_tensors.step_indices = torch.tensor(
                self.selected_step_indices, dtype=torch.long
            )
        self.stage._maybe_finalize_denoising_env_collection(
            self.batch, self.server_args.pipeline_config
        )
        if not self.batch.rollout_return_transition_pairs:
            return
        if len(self.selected_latents) != len(self.selected_step_indices):
            raise RuntimeError(
                "U1.5 rollout did not collect every selected transition"
            )
        if self.batch.rollout_trajectory_data is None:
            self.batch.rollout_trajectory_data = RolloutTrajectoryData()
        indices = torch.tensor(self.selected_step_indices, dtype=torch.long)
        self.batch.rollout_trajectory_data.transition_pairs = RolloutTransitionPairs(
            step_indices=indices,
            latents=torch.stack(self.selected_latents, dim=1),
            next_latents=torch.stack(self.selected_next_latents, dim=1),
            timesteps=self.timesteps[indices].cpu(),
            next_timesteps=self.timesteps[indices + 1].cpu(),
            sigmas=self.sigmas[indices].cpu(),
            next_sigmas=self.sigmas[indices + 1].cpu(),
            base_noise_scale=self.noise_scale,
        )


def _load_images(image_paths: str | Sequence[str] | None) -> list[Image.Image]:
    if image_paths is None:
        return []
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.getchannel("A"))
                images.append(background)
            else:
                images.append(image.convert("RGB"))
    return images


def _normalize_output(output: torch.Tensor) -> torch.Tensor:
    # U1.5 returns [-1, 1] BCHW tensors; SGLang image stages return
    # float tensors in [0, 1].
    return ((output.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def _resize_input_images(
    images: list[Image.Image], budget: int | str | None
) -> list[Image.Image]:
    if not images or budget is None:
        return images
    if budget == "auto":
        budget = (
            _DEFAULT_INPUT_PIXELS
            if len(images) <= 2
            else max(_MIN_INPUT_PIXELS, 2 * _DEFAULT_INPUT_PIXELS // len(images))
        )
    try:
        budget = int(budget)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_max_pixels must be an integer or 'auto'") from exc
    if budget < _MIN_INPUT_PIXELS:
        raise ValueError("input_max_pixels must be at least 512*512")

    resized = []
    for image in images:
        height, width = smart_resize(
            image.height,
            image.width,
            factor=32,
            min_pixels=budget,
            max_pixels=budget,
        )
        if (width, height) == image.size:
            resized.append(image)
            continue
        resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    return resized


class SenseNovaU1_5GenerationStage(PipelineStage, RolloutDenoisingMixin):
    """Run one non-thinking T2I or image-edit request."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    @staticmethod
    def _resolve_size(batch: Req, images: list[Image.Image]) -> tuple[int, int]:
        width, height = batch.width, batch.height
        explicit_fields = getattr(batch.sampling_params, "_explicit_fields", set())
        explicit_size = "width" in explicit_fields and "height" in explicit_fields
        if width is not None and height is not None and (explicit_size or not images):
            if width % 32 or height % 32:
                raise ValueError(
                    f"SenseNova-U1.5 requires width/height divisible by 32, got {width}x{height}"
                )
            return int(width), int(height)

        if images:
            target = int(getattr(batch, "target_pixels", 2048 * 2048))
            height, width = smart_resize(
                images[0].height,
                images[0].width,
                factor=32,
                min_pixels=target,
                max_pixels=target,
            )
            return width, height
        return int(width or 2048), int(height or 2048)

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.server_args = server_args
        prompts = (
            [batch.prompt]
            if isinstance(batch.prompt, str)
            else list(batch.prompt or [])
        )
        if not prompts:
            raise ValueError("SenseNova-U1.5 requires a non-empty prompt")
        images = _resize_input_images(
            _load_images(batch.image_path), getattr(batch, "input_max_pixels", "auto")
        )
        if batch.rollout:
            if not images:
                raise ValueError(
                    "SenseNova-U1.5 rollout currently supports image editing only"
                )
            if len(images) != 1:
                raise ValueError(
                    "SenseNova-U1.5 rollout requires exactly one input image"
                )
            if float(batch.img_cfg_scale) != 1.0:
                raise ValueError(
                    "SenseNova-U1.5 image-edit rollout requires img_cfg_scale=1"
                )
            if len(prompts) != 1:
                raise ValueError(
                    "SenseNova-U1.5 rollout accepts one prompt per request; use "
                    "num_outputs_per_prompt for the GRPO group"
                )
        width, height = self._resolve_size(batch, images)
        batch_size = int(batch.num_outputs_per_prompt)
        outputs = []

        raw_seed = batch.seed
        if isinstance(raw_seed, list):
            seed_values = [int(seed) for seed in raw_seed]
            base_seed = seed_values[0]
            if len(seed_values) != batch_size:
                seed_values = [base_seed + index for index in range(batch_size)]
        else:
            base_seed = int(raw_seed or 0)
            seed_values = [base_seed + index for index in range(batch_size)]
        generators = (
            [
                torch.Generator(device=self.model.device).manual_seed(seed)
                for seed in seed_values
            ]
            if batch.rollout
            else None
        )

        raw_image_paths = batch.image_path or []
        if isinstance(raw_image_paths, str):
            raw_image_paths = [raw_image_paths]

        for prompt in prompts:
            rollout_controller = (
                _SenseNovaU1_5RolloutController(
                    self,
                    batch,
                    server_args,
                    generators,
                    prompt=prompt,
                    image_paths=[str(path) for path in raw_image_paths],
                    preprocessed_image_sizes=[image.size for image in images],
                    output_size=(width, height),
                )
                if batch.rollout
                else None
            )
            with set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=batch
            ):
                if images:
                    result = self.model.it2i_generate(
                        self.tokenizer,
                        prompt,
                        images,
                        cfg_scale=float(batch.cfg_scale),
                        img_cfg_scale=float(batch.img_cfg_scale),
                        cfg_norm=str(batch.cfg_norm),
                        timestep_shift=float(batch.timestep_shift),
                        t_eps=float(batch.t_eps),
                        image_size=(width, height),
                        num_steps=int(batch.num_inference_steps),
                        batch_size=batch_size,
                        seed=base_seed,
                        generators=generators,
                        denoise_start_callback=(
                            rollout_controller.start if rollout_controller else None
                        ),
                        denoise_step_callback=(
                            rollout_controller.step if rollout_controller else None
                        ),
                    )
                else:
                    result = self.model.t2i_generate(
                        self.tokenizer,
                        prompt,
                        cfg_scale=float(batch.cfg_scale),
                        cfg_norm=str(batch.cfg_norm),
                        timestep_shift=float(batch.timestep_shift),
                        t_eps=float(batch.t_eps),
                        image_size=(width, height),
                        num_steps=int(batch.num_inference_steps),
                        batch_size=batch_size,
                        seed=base_seed,
                        generators=generators,
                        denoise_start_callback=(
                            rollout_controller.start if rollout_controller else None
                        ),
                        denoise_step_callback=(
                            rollout_controller.step if rollout_controller else None
                        ),
                    )
            if rollout_controller is not None:
                rollout_controller.finalize()
            outputs.append(_normalize_output(result))

        frames = torch.cat(outputs, dim=0)
        return OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            metrics=batch.metrics,
        )
