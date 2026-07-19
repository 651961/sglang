# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bernini-R visual conditioning and guided Wan2.2 denoising stages."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

import imageio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp import FSDPModule

try:
    import decord
except ImportError:  # pragma: no cover - imageio remains the supported fallback
    decord = None

from sglang.multimodal_gen.configs.sample.bernini import (
    BERNINI_GUIDANCE_MODES as BERNINI_FULL_GUIDANCE_MODES,
    BERNINI_R_GUIDANCE_MODES,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision
from sglang.multimodal_gen.runtime.utils.vision import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)

BERNINI_GUIDANCE_MODES = BERNINI_R_GUIDANCE_MODES - {"auto"}


@contextmanager
def _unshard_if_fsdp(module):
    is_fsdp = isinstance(module, FSDPModule)
    if is_fsdp:
        module.unshard()
    try:
        yield module
    finally:
        if is_fsdp:
            module.reshard()


def make_source_ids(num_sources: int, max_source_id: int = 5) -> list[float]:
    """Map source segments into Bernini-R's trained source-ID range."""
    if num_sources <= 0:
        return []
    if num_sources > max_source_id:
        return torch.linspace(1.0, float(max_source_id), num_sources).tolist()
    return [float(index) for index in range(1, num_sources + 1)]


_PatchedSegment = tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


def _patch_latent_segments(
    model,
    source_latents: list[torch.Tensor],
    target: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    max_source_id: int,
) -> tuple[list[_PatchedSegment], _PatchedSegment]:
    source_ids = make_source_ids(len(source_latents), max_source_id)
    with _unshard_if_fsdp(model):
        sources = [
            model.patch_vae_latent(latent.to(target_dtype), source_id=source_id)
            for latent, source_id in zip(source_latents, source_ids, strict=True)
        ]
        target_segment = model.patch_vae_latent(target, source_id=0)
    return sources, target_segment


def _pack_segments(segments: list[_PatchedSegment]) -> _PatchedSegment:
    return (
        torch.cat([tokens for tokens, _ in segments], dim=1),
        (
            torch.cat([rotary[0] for _, rotary in segments], dim=0),
            torch.cat([rotary[1] for _, rotary in segments], dim=0),
        ),
    )


def resolve_guidance_mode(
    requested_mode: str,
    *,
    has_video: bool,
    has_image: bool,
    num_frames: int,
    has_edit_image: bool,
) -> str:
    """Resolve ``auto`` to the route used by the official Bernini-R scripts."""
    if requested_mode != "auto":
        if requested_mode not in BERNINI_GUIDANCE_MODES:
            raise ValueError(f"Unsupported Bernini-R guidance mode: {requested_mode}")
        return requested_mode
    if has_video and has_image:
        return "rv2v"
    if has_video:
        return "v2v_apg"
    if has_image:
        # A scalar image with a one-frame target is image editing. Lists are
        # reference images, even if a one-frame output was requested.
        if has_edit_image and num_frames == 1:
            return "v2v"
        return "r2v_apg"
    return "t2v_apg"


@dataclass
class MomentumBuffer:
    momentum: float
    running_average: torch.Tensor | int = 0

    def update(self, value: torch.Tensor) -> torch.Tensor:
        self.running_average = value + self.momentum * self.running_average
        return self.running_average


def normalize_apg_difference(
    difference: torch.Tensor,
    conditional_prediction: torch.Tensor,
    *,
    momentum_buffer: MomentumBuffer | None,
    eta: float,
    norm_threshold: float,
) -> torch.Tensor:
    """Apply Bernini's adaptive projected-guidance normalization."""
    if momentum_buffer is not None:
        difference = momentum_buffer.update(difference)

    reduce_dims = (-1, -2, -4)  # channel, height, width for [B,C,T,H,W]
    if norm_threshold > 0:
        difference_norm = difference.norm(p=2, dim=reduce_dims, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(difference_norm),
            torch.as_tensor(
                norm_threshold, device=difference.device, dtype=difference.dtype
            )
            / difference_norm,
        )
        difference = difference * scale

    difference_double = difference.double()
    reference = F.normalize(conditional_prediction.double(), dim=reduce_dims)
    parallel = (difference_double * reference).sum(
        dim=reduce_dims, keepdim=True
    ) * reference
    orthogonal = difference_double - parallel
    # The reference casts each projection back before recombination.
    parallel = parallel.to(difference.dtype)
    orthogonal = orthogonal.to(difference.dtype)
    return orthogonal + eta * parallel


def normalized_guidance(
    conditional_prediction: torch.Tensor,
    unconditional_prediction: torch.Tensor,
    guidance_scale: float,
    *,
    momentum_buffer: MomentumBuffer | None,
    eta: float,
    norm_threshold: float,
) -> torch.Tensor:
    normalized_delta = normalize_apg_difference(
        conditional_prediction - unconditional_prediction,
        conditional_prediction,
        momentum_buffer=momentum_buffer,
        eta=eta,
        norm_threshold=norm_threshold,
    )
    return unconditional_prediction + guidance_scale * normalized_delta


def normalized_guidance_chain(
    unconditional_prediction: torch.Tensor,
    conditional_predictions: list[torch.Tensor],
    guidance_scales: list[float],
    *,
    momentum_buffers: list[MomentumBuffer | None],
    eta: float,
    norm_thresholds: list[float],
) -> torch.Tensor:
    if not (
        len(conditional_predictions)
        == len(guidance_scales)
        == len(momentum_buffers)
        == len(norm_thresholds)
    ):
        raise ValueError("Chained APG inputs must have matching lengths")
    bases = [unconditional_prediction, *conditional_predictions]
    result = unconditional_prediction
    for index, conditional_prediction in enumerate(conditional_predictions):
        normalized_delta = normalize_apg_difference(
            conditional_prediction - bases[index],
            conditional_prediction,
            momentum_buffer=momentum_buffers[index],
            eta=eta,
            norm_threshold=norm_thresholds[index],
        )
        result = result + guidance_scales[index] * normalized_delta
    return result


def _make_divisible(value: int, stride: int = 16) -> int:
    return max(stride, int(round(value / stride) * stride))


def _resize_image(
    image: Image.Image, max_image_size: int, min_image_size: int = 1
) -> Image.Image:
    """Match Bernini's no-crop, bicubic, max-long-edge resize."""
    image = image.convert("RGB")
    width, height = image.size
    scale = min(max_image_size / max(width, height), 1.0)
    scale = max(scale, min_image_size / min(width, height))
    new_width = _make_divisible(round(width * scale))
    new_height = _make_divisible(round(height * scale))
    if max(new_width, new_height) > max_image_size:
        scale = max_image_size / max(new_width, new_height)
        new_width = _make_divisible(round(new_width * scale))
        new_height = _make_divisible(round(new_height * scale))
    if (new_width, new_height) == (width, height):
        return image
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    return normalize(numpy_to_pt(pil_to_numpy(image)))[0]


def _smart_video_indices(
    total_frames: int,
    source_fps: float,
    target_fps: int,
    max_frames: int,
    *,
    frame_factor: int = 4,
    add_one: bool = True,
) -> list[int]:
    """Sample frames using the official Bernini temporal policy."""
    if total_frames <= 0:
        raise ValueError("The source video contains no frames")
    source_total_frames = total_frames
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = float(target_fps)

    num_frames = total_frames / source_fps * target_fps
    num_frames = math.floor(num_frames / frame_factor) * frame_factor + int(add_one)
    num_frames = max(num_frames, frame_factor + int(add_one))
    if source_fps == target_fps:
        total_frames = math.floor(total_frames / frame_factor) * frame_factor + int(
            add_one
        )
    indices = torch.linspace(0, total_frames - 1, num_frames).round().long()
    max_frames = math.floor(max_frames / frame_factor) * frame_factor + int(add_one)
    if indices.numel() > max_frames:
        indices = indices[:max_frames]
    # The official reader clamps after the 4k+1 adjustment (which can add one
    # virtual endpoint when the source length is exactly 4k). Clamp here before
    # asking imageio for that frame.
    return indices.clamp_(0, source_total_frames - 1).tolist()


def _as_path_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


class BerniniVisualConditioningStage(PipelineStage):
    """Preprocess and VAE-encode Bernini's independent visual source segments."""

    def __init__(self, vae, *, with_planner: bool = False) -> None:
        super().__init__()
        self.vae = vae
        self.with_planner = with_planner

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                target_dtype=resolve_precision(
                    server_args, "vae", precision_attr="vae_precision"
                ),
            )
        ]

    @staticmethod
    def _validate_runtime(batch: Req) -> None:
        if batch.batch_size != 1 or batch.num_outputs_per_prompt != 1:
            raise ValueError("Bernini currently supports batch size 1 and one output")
        if batch.enable_teacache:
            raise NotImplementedError(
                "TeaCache is not valid across Bernini guidance branches"
            )
        if batch.progressive_mode != "fullres":
            raise NotImplementedError(
                "Bernini does not yet support progressive resolution"
            )
        if batch.rollout:
            raise NotImplementedError("Bernini rollout is not implemented")

    @staticmethod
    def _read_video_frames(
        path: str,
        *,
        fps: int,
        max_frames: int,
        frame_factor: int = 4,
        add_one: bool = True,
    ) -> list[Image.Image]:
        if not os.path.isfile(path):
            raise ValueError(f"Bernini-R video path does not exist: {path}")
        if decord is not None:
            # Use the official reader when available, including its single
            # decoding thread and fault-tolerance policy.
            reader = decord.VideoReader(
                path, num_threads=1, ctx=decord.cpu(0), fault_tol=1
            )
            source_fps = float(reader.get_avg_fps())
            indices = _smart_video_indices(
                len(reader),
                source_fps,
                fps,
                max_frames,
                frame_factor=frame_factor,
                add_one=add_one,
            )
            selected = [
                Image.fromarray(frame).convert("RGB")
                for frame in reader.get_batch(indices).asnumpy()
            ]
        else:
            with imageio.get_reader(path) as reader:
                metadata = reader.get_meta_data()
                source_fps = float(metadata.get("fps") or fps)
                try:
                    total_frames = int(reader.count_frames())
                except (RuntimeError, ValueError, AttributeError, OverflowError):
                    raw_total_frames = metadata.get("nframes") or 0
                    try:
                        numeric_total_frames = float(raw_total_frames)
                        total_frames = (
                            int(numeric_total_frames)
                            if math.isfinite(numeric_total_frames)
                            else 0
                        )
                    except (TypeError, ValueError, OverflowError):
                        total_frames = 0
                if total_frames <= 0:
                    # Only used for containers without a usable frame count.
                    frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
                    total_frames = len(frames)
                    indices = _smart_video_indices(
                        total_frames,
                        source_fps,
                        fps,
                        max_frames,
                        frame_factor=frame_factor,
                        add_one=add_one,
                    )
                    selected = [frames[index] for index in indices]
                else:
                    indices = _smart_video_indices(
                        total_frames,
                        source_fps,
                        fps,
                        max_frames,
                        frame_factor=frame_factor,
                        add_one=add_one,
                    )
                    selected = [
                        Image.fromarray(reader.get_data(index)).convert("RGB")
                        for index in indices
                    ]

        return selected

    @classmethod
    def _read_video(
        cls,
        path: str,
        *,
        fps: int,
        max_image_size: int,
        min_image_size: int = 1,
        max_frames: int,
    ) -> torch.Tensor:
        selected = cls._read_video_frames(path, fps=fps, max_frames=max_frames)
        processed = [
            _image_to_tensor(_resize_image(frame, max_image_size, min_image_size))
            for frame in selected
        ]
        first_size = processed[0].shape[-2:]
        if any(frame.shape[-2:] != first_size for frame in processed):
            raise ValueError(
                "All frames in a source video must have the same resolution"
            )
        return torch.stack(processed, dim=1).unsqueeze(0)

    @staticmethod
    def _read_image(
        path: str, *, max_image_size: int, min_image_size: int = 1
    ) -> torch.Tensor:
        image = load_image(path)
        image = _resize_image(image, max_image_size, min_image_size)
        return _image_to_tensor(image).unsqueeze(0).unsqueeze(2)

    @staticmethod
    def _encode(vae, pixels: torch.Tensor, server_args: ServerArgs) -> torch.Tensor:
        device = get_local_torch_device()
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        pixels = pixels.to(device=device, dtype=vae_dtype)
        encoded = vae.encode(pixels)
        latent_dist = getattr(encoded, "latent_dist", encoded)
        latents = latent_dist.mode()

        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                latents.device, latents.dtype, vae
            )
        )
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                shift_factor = shift_factor.to(latents.device, latents.dtype)
            latents = latents - shift_factor
        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latents.device, latents.dtype)
        latents = latents * scaling_factor
        return latents.float()

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self._validate_runtime(batch)
        min_image_size = 240 if self.with_planner else 1

        video_paths = _as_path_list(batch.video_path)
        edit_image_path = (
            batch.image_path if isinstance(batch.image_path, str) else None
        )
        image_paths = (
            [] if edit_image_path is not None else _as_path_list(batch.image_path)
        )
        image_paths.extend(_as_path_list(batch.reference_image_paths))

        videos = [
            self._read_video(
                path,
                fps=batch.fps,
                max_image_size=batch.max_image_size,
                min_image_size=min_image_size,
                max_frames=batch.num_frames,
            )
            for path in video_paths
        ]
        edit_image = (
            self._read_image(
                edit_image_path,
                max_image_size=batch.max_image_size,
                min_image_size=min_image_size,
            )
            if edit_image_path is not None
            else None
        )
        images = [] if edit_image is None else [edit_image]
        images.extend(
            self._read_image(
                path,
                max_image_size=batch.max_image_size,
                min_image_size=min_image_size,
            )
            for path in image_paths
        )

        if videos:
            batch.num_frames = videos[0].shape[2]
            batch.height, batch.width = videos[0].shape[-2:]
        elif edit_image is not None and not {
            "height",
            "width",
        }.issubset(batch.extra.get("explicit_fields", ())):
            batch.height, batch.width = edit_image.shape[-2:]
        batch.num_frames = server_args.pipeline_config.adjust_num_frames(
            batch.num_frames
        )
        batch.height = _make_divisible(int(batch.height))
        batch.width = _make_divisible(int(batch.width))

        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            self.vae = vae
            video_latents = [self._encode(vae, video, server_args) for video in videos]
            image_latents = [self._encode(vae, image, server_args) for image in images]

        batch.extra["bernini_video_latents"] = video_latents
        batch.extra["bernini_image_latents"] = image_latents
        if self.with_planner:
            mode = batch.guidance_mode
            if mode == "auto":
                mode = (
                    "rv2v_wapg"
                    if video_latents and image_latents
                    else "vae_txt_vit_wapg"
                )
            if mode not in BERNINI_FULL_GUIDANCE_MODES - {"auto"}:
                raise ValueError(f"Unsupported Bernini guidance mode: {mode}")
            batch.extra["bernini_guidance_mode"] = mode
        else:
            batch.extra["bernini_guidance_mode"] = resolve_guidance_mode(
                batch.guidance_mode,
                has_video=bool(video_latents),
                has_image=bool(image_latents),
                num_frames=batch.num_frames,
                has_edit_image=edit_image is not None,
            )
        return batch


class BerniniTimestepPreparationStage(TimestepPreparationStage):
    """Apply a request-level flow shift before building the UniPC schedule."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.scheduler is not None and batch.timesteps is not None:
            return batch

        scheduler = get_or_create_request_scheduler(batch, self.scheduler, isolate=True)
        flow_shift = batch.flow_shift
        if flow_shift is None:
            flow_shift = server_args.pipeline_config.flow_shift
        if flow_shift is not None:
            flow_shift = float(flow_shift)
            configured_shift = getattr(
                getattr(scheduler, "config", None), "flow_shift", None
            )
            if configured_shift is None or not math.isclose(
                float(configured_shift), flow_shift
            ):
                if hasattr(scheduler, "set_shift"):
                    scheduler.set_shift(flow_shift)
                elif hasattr(scheduler.__class__, "from_config"):
                    scheduler = scheduler.__class__.from_config(
                        scheduler.config, flow_shift=flow_shift
                    )
                else:
                    raise TypeError(
                        f"{scheduler.__class__.__name__} cannot apply Bernini-R "
                        "flow_shift overrides"
                    )
                batch.scheduler = scheduler
        return super().forward(batch, server_args)


class BerniniRDenoisingStage(DenoisingStage):
    """Wan2.2 dual-expert denoising with Bernini source-prefix guidance."""

    def _cache_dit_dual_model_name(self) -> str:
        return "bernini-r"

    @staticmethod
    def _thresholds(batch: Req, count: int) -> list[float]:
        raw = batch.norm_threshold
        values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        if not values:
            raise ValueError("norm_threshold must not be empty")
        if len(values) == 1:
            values *= count
        if len(values) < count:
            values.extend([values[-1]] * (count - len(values)))
        return [float(value) for value in values[:count]]

    def _before_denoising_loop(
        self, ctx: DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        super()._before_denoising_loop(ctx, batch, server_args)
        mode = batch.extra["bernini_guidance_mode"]
        momentum = float(batch.momentum)
        if mode == "r2v_apg":
            ctx.extra["bernini_momentum_buffers"] = [
                MomentumBuffer(momentum),
                MomentumBuffer(momentum),
            ]
        elif mode in {"v2v_apg", "t2v_apg"}:
            ctx.extra["bernini_momentum_buffers"] = [MomentumBuffer(momentum)]
        else:
            ctx.extra["bernini_momentum_buffers"] = []

    @staticmethod
    def _validate_mode_sources(
        mode: str, video_latents: list[torch.Tensor], image_latents: list[torch.Tensor]
    ) -> None:
        if mode == "rv2v" and not (video_latents and image_latents):
            raise ValueError("rv2v guidance requires a video and at least one image")
        if mode == "v2v_chain" and not video_latents:
            raise ValueError("v2v_chain guidance requires a source video")
        if mode in {"v2v", "v2v_apg"} and not (video_latents or image_latents):
            raise ValueError(f"{mode} guidance requires a visual source")
        if mode == "r2v_apg" and not image_latents:
            raise ValueError("r2v_apg guidance requires at least one reference image")

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        mode = batch.extra["bernini_guidance_mode"]
        video_latents = batch.extra["bernini_video_latents"]
        image_latents = batch.extra["bernini_image_latents"]
        self._validate_mode_sources(mode, video_latents, image_latents)

        current_model = step.current_model
        target = ctx.scheduler.scale_model_input(ctx.latents, step.t_device)
        target = target.to(ctx.target_dtype)

        source_latents = (
            image_latents if mode == "r2v_apg" else [*video_latents, *image_latents]
        )
        source_segments, target_segment = _patch_latent_segments(
            current_model,
            source_latents,
            target,
            target_dtype=ctx.target_dtype,
            max_source_id=server_args.pipeline_config.max_source_id,
        )
        combination_segments = {"none": []}
        if mode in {"rv2v", "v2v", "v2v_chain", "v2v_apg"}:
            combination_segments["vi"] = source_segments
            if mode in {"rv2v", "v2v_chain"}:
                combination_segments["v"] = source_segments[:1]
        elif mode == "r2v_apg":
            combination_segments["i"] = source_segments
        target_length = target_segment[0].shape[1]

        def assemble(prefix):
            return _pack_segments([*prefix, target_segment])

        combinations = {}

        positive_text = ctx.pos_cond_kwargs["encoder_hidden_states"].to(
            ctx.target_dtype
        )
        negative_text = ctx.neg_cond_kwargs.get("encoder_hidden_states")
        if negative_text is None:
            raise ValueError("Bernini-R requires negative prompt embeddings")
        negative_text = negative_text.to(ctx.target_dtype)
        timestep = step.t_device.expand(target.shape[0])

        cache_branches: dict[tuple[str, bool], int] = {}

        def predict(combo: str, text: torch.Tensor) -> torch.Tensor:
            if combo not in combinations:
                combinations[combo] = assemble(combination_segments[combo])
            packed, rotary = combinations[combo]
            branch_key = (combo, text is positive_text)
            cache_branch = cache_branches.setdefault(branch_key, len(cache_branches))
            with set_forward_context(
                current_timestep=step.step_index,
                attn_metadata=None,
                forward_batch=batch,
            ):
                prediction = current_model(
                    hidden_states=packed,
                    encoder_hidden_states=text,
                    timestep=timestep,
                    packed_rotary_emb=rotary,
                    packed_cache_branch=cache_branch,
                )
            target_prediction = prediction[:, -target_length:, :]
            return current_model.unpatchify(target_prediction, ctx.latents.shape)

        scale_multiplier = (
            float(batch.omega_scale)
            if ctx.boundary_timestep is not None and step.t_int < ctx.boundary_timestep
            else 1.0
        )
        omega_video = float(batch.omega_vid) * scale_multiplier
        omega_image = float(batch.omega_img) * scale_multiplier
        omega_text = float(batch.omega_txt) * scale_multiplier

        if mode == "rv2v":
            unconditional = predict("none", negative_text)
            video_unconditional = predict("v", negative_text)
            visual_unconditional = predict("vi", negative_text)
            conditional = predict("vi", positive_text)
            noise_prediction = (
                unconditional
                + omega_video * (video_unconditional - unconditional)
                + omega_image * (visual_unconditional - video_unconditional)
                + omega_text * (conditional - visual_unconditional)
            )
        elif mode == "v2v":
            unconditional = predict("vi", negative_text)
            conditional = predict("vi", positive_text)
            noise_prediction = unconditional + omega_text * (
                conditional - unconditional
            )
        elif mode == "v2v_chain":
            unconditional = predict("none", negative_text)
            video_unconditional = predict("v", negative_text)
            conditional = predict("vi", positive_text)
            noise_prediction = (
                unconditional
                + omega_video * (video_unconditional - unconditional)
                + omega_text * (conditional - video_unconditional)
            )
        elif mode == "t2v":
            unconditional = predict("none", negative_text)
            conditional = predict("none", positive_text)
            noise_prediction = unconditional + omega_text * (
                conditional - unconditional
            )
        else:
            scheduler_index = (
                0 if ctx.scheduler.step_index is None else ctx.scheduler.step_index
            )
            sigma = ctx.scheduler.sigmas[scheduler_index].to(
                device=ctx.latents.device, dtype=ctx.latents.dtype
            )
            noisy = ctx.latents
            eta = float(batch.eta)
            buffers = ctx.extra["bernini_momentum_buffers"]

            if mode == "r2v_apg":
                unconditional_flow = predict("none", negative_text)
                image_flow = predict("i", negative_text)
                conditional_flow = predict("i", positive_text)
                unconditional_x0 = noisy - sigma * unconditional_flow
                image_x0 = noisy - sigma * image_flow
                conditional_x0 = noisy - sigma * conditional_flow
                guided_x0 = normalized_guidance_chain(
                    unconditional_x0,
                    [image_x0, conditional_x0],
                    [omega_image, omega_text],
                    momentum_buffers=buffers,
                    eta=eta,
                    norm_thresholds=self._thresholds(batch, 2),
                )
            elif mode in {"v2v_apg", "t2v_apg"}:
                source = "vi" if mode == "v2v_apg" else "none"
                unconditional_flow = predict(source, negative_text)
                conditional_flow = predict(source, positive_text)
                unconditional_x0 = noisy - sigma * unconditional_flow
                conditional_x0 = noisy - sigma * conditional_flow
                guided_x0 = normalized_guidance(
                    conditional_x0,
                    unconditional_x0,
                    omega_text,
                    momentum_buffer=buffers[0],
                    eta=eta,
                    norm_threshold=self._thresholds(batch, 1)[0],
                )
            else:  # guarded by the sampling config and resolve_guidance_mode
                raise ValueError(f"Unsupported Bernini-R guidance mode: {mode}")
            noise_prediction = (noisy - guided_x0) / sigma

        if server_args.comfyui_mode:
            batch.noise_pred = noise_prediction
        ctx.latents = ctx.scheduler.step(
            model_output=noise_prediction,
            timestep=step.t_device,
            sample=ctx.latents,
            **ctx.extra_step_kwargs,
            return_dict=False,
        )[0]
