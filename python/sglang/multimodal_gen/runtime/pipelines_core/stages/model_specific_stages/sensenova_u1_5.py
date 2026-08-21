# SPDX-License-Identifier: Apache-2.0
"""Single-stage non-thinking SenseNova-U1.5 generation/editing stage."""

from __future__ import annotations

from typing import Sequence

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
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.multimodal.processors.qwen_vl import smart_resize

_DEFAULT_INPUT_PIXELS = 2048 * 2048
_MIN_INPUT_PIXELS = 512 * 512


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


class SenseNovaU1_5GenerationStage(PipelineStage):
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
        width, height = self._resolve_size(batch, images)
        batch_size = int(batch.num_outputs_per_prompt)
        outputs = []

        for prompt in prompts:
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
                        seed=int(
                            batch.seed[0]
                            if isinstance(batch.seed, list)
                            else batch.seed
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
                        seed=int(
                            batch.seed[0]
                            if isinstance(batch.seed, list)
                            else batch.seed
                        ),
                    )
            outputs.append(_normalize_output(result))

        frames = torch.cat(outputs, dim=0)
        return OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            metrics=batch.metrics,
        )
