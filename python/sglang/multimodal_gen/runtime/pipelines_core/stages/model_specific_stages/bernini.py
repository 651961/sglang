# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Semantic planning and renderer guidance for Bernini."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.configs.sample.bernini import (
    BERNINI_R_SYSTEM_PROMPTS,
    bernini_prompt_clean,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bernini_r import (
    BerniniVisualConditioningStage,
    _as_path_list,
    _pack_segments,
    _patch_latent_segments,
    _smart_video_indices,
    _unshard_if_fsdp,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.vision import load_image

_VIT_MIN_PIXELS = 3136
_VIT_MAX_PIXELS = 50176
_MAX_VISUAL_ITEMS = 64


@dataclass
class _VisualItem:
    kind: str
    grid_thw: torch.Tensor
    features: torch.Tensor | None


@dataclass
class _PlannerBranch:
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    output_mask: torch.Tensor


def _build_attention_mask(
    token_types: torch.Tensor, segment_ids: torch.Tensor
) -> torch.Tensor:
    """Bernini's causal-text, bidirectional-visual attention mask."""
    query_types = token_types[:, :, None]
    key_types = token_types[:, None, :]
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = torch.ones(
        token_types.shape[1],
        token_types.shape[1],
        dtype=torch.bool,
        device=token_types.device,
    ).tril_()[None]
    visible_text_or_input = causal & ((key_types == 0) | (key_types == 2))
    visible = (
        (((query_types == 0) | (query_types == 2)) & visible_text_or_input)
        | (
            (query_types == 1)
            & (visible_text_or_input | ((key_types == 1) & same_segment))
        )
        | (
            (query_types == 3)
            & (visible_text_or_input | ((key_types == 3) & same_segment))
        )
    )
    mask = torch.zeros_like(visible, dtype=torch.float32)
    return mask.masked_fill_(~visible, float("-inf"))


def _apg_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    parallel_scale: float = 0.2,
) -> torch.Tensor:
    batch_size = delta.shape[0]
    delta_flat = delta.reshape(batch_size, -1)
    reference_flat = reference.reshape(batch_size, -1)
    norm_sq = reference_flat.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    parallel = (
        (delta_flat * reference_flat).sum(dim=1, keepdim=True)
        / norm_sq
        * reference_flat
    )
    return (delta_flat - parallel + parallel_scale * parallel).reshape_as(delta)


class BerniniPlanningStage(PipelineStage):
    """Run Qwen2.5-VL planning and produce the four renderer contexts."""

    def __init__(self, planner, processor, t5_encoder, t5_tokenizer) -> None:
        super().__init__()
        self.planner = planner
        self.processor = processor
        self.t5_encoder = t5_encoder
        self.t5_tokenizer = t5_tokenizer

        tokenizer = processor.tokenizer
        self.input_tokens = [
            f"<|visual_input_token_pad_{index}|>" for index in range(_MAX_VISUAL_ITEMS)
        ]
        self.output_tokens = [
            f"<|visual_output_token_pad_{index}|>" for index in range(_MAX_VISUAL_ITEMS)
        ]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": self.input_tokens + self.output_tokens}
        )
        self.input_token_ids = tokenizer.convert_tokens_to_ids(self.input_tokens)
        self.output_token_ids = tokenizer.convert_tokens_to_ids(self.output_tokens)
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "text_encoder", memory_intensive=True),
            ComponentUse(stage_name, "text_encoder_2", memory_intensive=True),
        ]

    @staticmethod
    def _paths(batch: Req) -> tuple[list[str], list[str]]:
        videos = _as_path_list(batch.video_path)
        images = _as_path_list(batch.image_path)
        images.extend(_as_path_list(batch.reference_image_paths))
        return videos, images

    def _preprocess_visuals(
        self, batch: Req
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        tuple[str, torch.Tensor],
    ]:
        video_paths, image_paths = self._paths(batch)
        vit_fps = max(int(batch.fps) // 8, 1)

        videos = []
        for path in video_paths:
            frames = BerniniVisualConditioningStage._read_video_frames(
                path,
                fps=vit_fps,
                max_frames=batch.num_frames,
                frame_factor=2,
                add_one=False,
            )
            processed = self.processor.video_processor(
                videos=frames,
                return_tensors="pt",
                size={
                    "shortest_edge": _VIT_MIN_PIXELS,
                    "longest_edge": _VIT_MAX_PIXELS,
                },
            )
            videos.append((processed.pixel_values_videos, processed.video_grid_thw))

        images = []
        if image_paths:
            processed = self.processor.image_processor(
                images=[load_image(path) for path in image_paths],
                return_tensors="pt",
                min_pixels=_VIT_MIN_PIXELS,
                max_pixels=_VIT_MAX_PIXELS,
            )
            images.append((processed.pixel_values, processed.image_grid_thw))

        if batch.num_frames == 1:
            target_image = Image.new("RGB", (batch.width, batch.height))
            target = self.processor.image_processor(
                images=[target_image],
                return_tensors="pt",
                min_pixels=_VIT_MIN_PIXELS,
                max_pixels=_VIT_MAX_PIXELS,
            )
            target_spec = ("image", target.image_grid_thw[0])
        else:
            indices = _smart_video_indices(
                batch.num_frames,
                float(batch.fps),
                vit_fps,
                batch.num_frames,
                frame_factor=2,
                add_one=False,
            )
            target_frames = [
                Image.new("RGB", (batch.width, batch.height)) for _ in indices
            ]
            target = self.processor.video_processor(
                videos=target_frames,
                return_tensors="pt",
                size={
                    "shortest_edge": _VIT_MIN_PIXELS,
                    "longest_edge": _VIT_MAX_PIXELS,
                },
            )
            target_spec = ("video", target.video_grid_thw[0])
        return videos, images, target_spec

    @staticmethod
    def _visual_token_count(grid_thw: torch.Tensor, merge_size: int) -> int:
        return int(grid_thw.prod().item()) // merge_size**2

    @staticmethod
    def _pattern(token: str, count: int) -> str:
        return "<|vision_start|>" + token * count + "<|vision_end|>"

    def _build_branch(
        self,
        planner,
        source_items: list[_VisualItem],
        target_item: _VisualItem,
        *,
        task: str,
        prompt: str,
        include_sources: bool,
    ) -> _PlannerBranch:
        tokenizer = self.processor.tokenizer
        merge_size = self.processor.image_processor.merge_size
        if len(source_items) + 1 > _MAX_VISUAL_ITEMS:
            raise ValueError(
                f"Bernini supports at most {_MAX_VISUAL_ITEMS - 1} visual sources"
            )

        user_content = ""
        if include_sources:
            for visual_id, item in enumerate(source_items):
                count = self._visual_token_count(item.grid_thw, merge_size)
                user_content += self._pattern(self.input_tokens[visual_id], count)
        user_content += prompt

        target_id = len(source_items)
        target_count = self._visual_token_count(target_item.grid_thw, merge_size)
        target_content = self._pattern(self.output_tokens[target_id], target_count)
        system_prompt = BERNINI_R_SYSTEM_PROMPTS.get(
            task, BERNINI_R_SYSTEM_PROMPTS["default"]
        )

        input_ids = []
        for role, content in (
            ("system", system_prompt),
            ("user", user_content),
            ("assistant", target_content),
        ):
            content = content.strip()
            if not content:
                continue
            input_ids.extend(
                tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
            )
            input_ids.extend(tokenizer.encode(content, add_special_tokens=False))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        token_types = torch.zeros_like(input_ids)
        segment_ids = torch.arange(input_ids.numel())
        visual_input_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for visual_id, item in enumerate(source_items):
            if not include_sources:
                break
            mask = input_ids == self.input_token_ids[visual_id]
            token_types[mask] = 2
            segment_ids[mask] = visual_id + 1
            visual_input_mask |= mask
            input_ids[mask] = (
                self.image_pad_id if item.kind == "image" else self.video_pad_id
            )

        output_mask = input_ids == self.output_token_ids[target_id]
        if int(output_mask.sum()) != target_count:
            raise RuntimeError(
                "Bernini target visual-token count changed in tokenization"
            )
        token_types[output_mask] = 3
        segment_ids[output_mask] = target_id + 1
        input_ids[output_mask] = (
            self.image_pad_id if target_item.kind == "image" else self.video_pad_id
        )

        included_items = source_items if include_sources else []
        image_grids = [
            item.grid_thw
            for item in [*included_items, target_item]
            if item.kind == "image"
        ]
        video_grids = [
            item.grid_thw
            for item in [*included_items, target_item]
            if item.kind == "video"
        ]
        position_ids = planner.model.get_rope_index(
            input_ids=input_ids[None],
            image_grid_thw=torch.stack(image_grids) if image_grids else None,
            video_grid_thw=torch.stack(video_grids) if video_grids else None,
            attention_mask=torch.ones_like(input_ids)[None],
        )[0]

        device = planner.device
        input_ids = input_ids.to(device)
        inputs_embeds = planner.model.get_input_embeddings()(input_ids)
        visual_features = [
            item.features for item in included_items if item.features is not None
        ]
        visual_features.append(
            planner.mask_tokens[:, :1]
            .expand(1, target_count, -1)
            .reshape(target_count, -1)
        )
        combined_features = torch.cat(visual_features).to(inputs_embeds.dtype)
        visual_mask = (visual_input_mask | output_mask).to(device)
        if int(visual_mask.sum()) != combined_features.shape[0]:
            raise RuntimeError(
                "Bernini visual features do not match placeholder tokens"
            )
        inputs_embeds[visual_mask] = combined_features

        attention_mask = _build_attention_mask(token_types[None], segment_ids[None])[
            :, None
        ]
        return _PlannerBranch(
            inputs_embeds=inputs_embeds[None],
            position_ids=position_ids.to(device),
            attention_mask=attention_mask.to(device=device, dtype=inputs_embeds.dtype),
            output_mask=output_mask.to(device),
        )

    @staticmethod
    def _hidden_states(planner, branch: _PlannerBranch) -> torch.Tensor:
        return planner.hidden_states(
            branch.inputs_embeds,
            branch.position_ids,
            branch.attention_mask,
        )

    def _plan(
        self,
        planner,
        conditional: _PlannerBranch,
        unconditional: _PlannerBranch,
        image_conditional: _PlannerBranch,
        batch: Req,
    ) -> dict[str, torch.Tensor]:
        token_count = int(conditional.output_mask.sum())
        if token_count != int(unconditional.output_mask.sum()) or token_count != int(
            image_conditional.output_mask.sum()
        ):
            raise RuntimeError("Bernini planner branches have different target sizes")

        seed = int(batch.seeds[0] if batch.seeds else batch.seed)
        rng = np.random.RandomState(seed)
        # The official preprocessing shuffles an all-masked target once for
        # each of its three branches before drawing the MaskGIT order.
        for _ in range(3):
            rng.permutation(token_count)
        order = torch.from_numpy(rng.permutation(token_count)).to(
            conditional.inputs_embeds.device
        )
        mask = torch.ones(token_count, dtype=torch.bool, device=order.device)
        # Official Bernini reseeds the renderer's latent generator separately,
        # so planning must not advance batch.generator.
        generator = torch.Generator(device="cpu").manual_seed(seed)

        for step in self.progress_bar(
            range(batch.planning_step), desc="Bernini planning", batch=batch
        ):
            hidden_states = [
                self._hidden_states(planner, branch)
                for branch in (conditional, unconditional, image_conditional)
            ]
            predictions = [
                planner.connector["pred_vit"](hidden[:, branch.output_mask])
                for hidden, branch in zip(
                    hidden_states,
                    (conditional, unconditional, image_conditional),
                    strict=True,
                )
            ]

            ratio = math.cos(math.pi / 2 * (step + 1) / batch.planning_step)
            next_count = min(int(mask.sum()) - 1, math.floor(token_count * ratio))
            next_count = max(next_count, 1)
            next_mask = torch.zeros_like(mask)
            next_mask[order[:next_count]] = True
            predict_mask = mask if step + 1 == batch.planning_step else mask ^ next_mask
            mask = next_mask
            if not bool(predict_mask.any()):
                continue

            selected = torch.cat(
                [prediction[:, predict_mask] for prediction in predictions], dim=1
            )[0]
            generated = planner.vit_decoder.sample(
                selected,
                txt_cfg=float(batch.vit_txt_cfg),
                image_cfg=float(batch.vit_img_cfg),
                steps=int(batch.vit_denoising_step),
                generator=generator,
            )[: int(predict_mask.sum())]
            target_features = conditional.inputs_embeds[:, conditional.output_mask]
            target_features[:, predict_mask] = generated
            for branch in (conditional, unconditional, image_conditional):
                branch.inputs_embeds[:, branch.output_mask] = target_features

        conditional_hidden = self._hidden_states(planner, conditional)
        unconditional_hidden = self._hidden_states(planner, unconditional)
        conditional_context = planner.connector["proj_gen"](conditional_hidden)
        unconditional_context = planner.connector["proj_gen"](unconditional_hidden)
        return {
            "wtxt_wvit": conditional_context,
            "wtxt_wovit": conditional_context[:, ~conditional.output_mask],
            "wotxt_wvit": conditional_context[:, conditional.output_mask],
            "wotxt_wovit": unconditional_context[:, ~unconditional.output_mask],
        }

    def _encode_t5(self, encoder, text: str) -> torch.Tensor:
        tokens = self.t5_tokenizer(
            bernini_prompt_clean(text),
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        device = next(encoder.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        hidden_states = encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return hidden_states[:, : int(attention_mask.sum())]

    @staticmethod
    def _pad_context(context: torch.Tensor, length: int) -> torch.Tensor:
        if context.shape[1] >= length:
            return context
        return F.pad(context, (0, 0, 0, length - context.shape[1]))

    @torch.inference_mode()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        video_inputs, image_inputs, target_spec = self._preprocess_visuals(batch)
        with (
            self.use_declared_component(
                component_name="text_encoder", module=self.planner
            ) as planner,
            set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=batch
            ),
        ):
            assert planner is not None
            self.planner = planner
            with _unshard_if_fsdp(planner):
                source_items = []
                for pixel_values, grid_thw in video_inputs:
                    features = planner.visual_features(pixel_values, grid_thw)
                    source_items.extend(
                        _VisualItem("video", grid, feature)
                        for grid, feature in zip(grid_thw, features, strict=True)
                    )
                for pixel_values, grid_thw in image_inputs:
                    features = planner.visual_features(pixel_values, grid_thw)
                    source_items.extend(
                        _VisualItem("image", grid, feature)
                        for grid, feature in zip(grid_thw, features, strict=True)
                    )
                target_item = _VisualItem(target_spec[0], target_spec[1], None)

                task = batch.extra["bernini_task_type"]
                prompt = batch.extra["bernini_raw_prompt"]
                conditional = self._build_branch(
                    planner,
                    source_items,
                    target_item,
                    task=task,
                    prompt=prompt,
                    include_sources=True,
                )
                unconditional = self._build_branch(
                    planner,
                    source_items,
                    target_item,
                    task=task,
                    prompt=(
                        batch.negative_prompt if len(batch.negative_prompt) > 1 else ""
                    ),
                    include_sources=False,
                )
                image_conditional = self._build_branch(
                    planner,
                    source_items,
                    target_item,
                    task=task,
                    prompt=prompt,
                    include_sources=False,
                )
                contexts = self._plan(
                    planner, conditional, unconditional, image_conditional, batch
                )

        with self.use_declared_component(
            component_name="text_encoder_2", module=self.t5_encoder
        ) as t5_encoder:
            assert t5_encoder is not None
            self.t5_encoder = t5_encoder
            positive_t5 = self._encode_t5(t5_encoder, batch.prompt)
            negative_t5 = self._encode_t5(t5_encoder, batch.negative_prompt)

        contexts["wtxt_wvit"] = torch.cat((positive_t5, contexts["wtxt_wvit"]), dim=1)
        contexts["wtxt_wovit"] = torch.cat((positive_t5, contexts["wtxt_wovit"]), dim=1)
        contexts["wotxt_wvit"] = torch.cat((negative_t5, contexts["wotxt_wvit"]), dim=1)
        contexts["wotxt_wovit"] = torch.cat(
            (negative_t5, contexts["wotxt_wovit"]), dim=1
        )
        minimum_length = int(batch.max_sequence_length or 0)
        contexts = {
            name: self._pad_context(context, minimum_length)
            for name, context in contexts.items()
        }
        batch.extra["bernini_contexts"] = contexts
        batch.prompt_embeds = [contexts["wtxt_wvit"]]
        batch.negative_prompt_embeds = [contexts["wotxt_wovit"]]
        return batch


class BerniniDenoisingStage(DenoisingStage):
    """Multi-context renderer guidance from the Bernini planner."""

    def _cache_dit_dual_model_name(self) -> str:
        return "bernini"

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
        if mode == "rv2v_wapg" and not (video_latents and image_latents):
            raise ValueError("rv2v_wapg requires a video and at least one image")

        current_model = step.current_model
        target = ctx.scheduler.scale_model_input(ctx.latents, step.t_device).to(
            ctx.target_dtype
        )
        source_latents = [*video_latents, *image_latents]
        source_segments, target_segment = _patch_latent_segments(
            current_model,
            source_latents,
            target,
            target_dtype=ctx.target_dtype,
            max_source_id=server_args.pipeline_config.max_source_id,
        )

        combinations = {
            "none": [],
            "video": source_segments[: len(video_latents)],
            "all": source_segments,
        }
        target_length = target_segment[0].shape[1]
        packed_inputs = {}

        def assemble(name: str):
            if name not in packed_inputs:
                packed_inputs[name] = _pack_segments(
                    [*combinations[name], target_segment]
                )
            return packed_inputs[name]

        contexts = {
            name: context.to(ctx.target_dtype)
            for name, context in batch.extra["bernini_contexts"].items()
        }
        timestep = step.t_device.expand(target.shape[0])
        cache_branches = {}

        def predict(combo: str, context_name: str) -> torch.Tensor:
            packed, rotary = assemble(combo)
            branch_key = (combo, context_name)
            cache_branch = cache_branches.setdefault(branch_key, len(cache_branches))
            with set_forward_context(
                current_timestep=step.step_index,
                attn_metadata=None,
                forward_batch=batch,
            ):
                prediction = current_model(
                    hidden_states=packed,
                    encoder_hidden_states=contexts[context_name],
                    timestep=timestep,
                    packed_rotary_emb=rotary,
                    packed_cache_branch=cache_branch,
                )
            return current_model.unpatchify(
                prediction[:, -target_length:, :], ctx.latents.shape
            )

        multiplier = (
            float(batch.omega_scale)
            if ctx.boundary_timestep is not None and step.t_int < ctx.boundary_timestep
            else 1.0
        )
        omega_video = float(batch.omega_vid) * multiplier
        omega_image = float(batch.omega_img) * multiplier
        omega_text = float(batch.omega_txt) * multiplier
        omega_target = float(batch.omega_tgt) * multiplier

        baseline = predict("none", "wotxt_wovit")
        if mode == "rv2v_wapg":
            video = predict("video", "wotxt_wovit") if omega_video > 0 else baseline
            visual = predict("all", "wotxt_wovit") if omega_image > 0 else video
            text = predict("all", "wtxt_wovit") if omega_text > 0 else visual
            target_condition = predict("all", "wtxt_wvit") if omega_target > 0 else text
            noise_prediction = (
                baseline
                + omega_video * (video - baseline)
                + omega_image * (visual - video)
                + omega_text * (text - visual)
                + omega_target * (target_condition - text)
            )
        elif mode in {"vae_txt_vit", "vae_txt_vit_wapg"}:
            visual = (
                predict("all", "wotxt_wovit")
                if omega_image > 0 and source_segments
                else baseline
            )
            text = predict("all", "wtxt_wovit") if omega_text > 0 else visual
            target_condition = predict("all", "wtxt_wvit") if omega_target > 0 else text
            deltas = (
                visual - baseline,
                text - visual,
                target_condition - text,
            )
            if mode.endswith("_wapg"):
                deltas = tuple(
                    _apg_delta(delta, reference)
                    for delta, reference in zip(
                        deltas,
                        (visual, text, target_condition),
                        strict=True,
                    )
                )
            noise_prediction = (
                baseline
                + omega_image * deltas[0]
                + omega_text * deltas[1]
                + omega_target * deltas[2]
            )
        else:
            raise ValueError(f"Unsupported Bernini guidance mode: {mode}")

        if server_args.comfyui_mode:
            batch.noise_pred = noise_prediction
        ctx.latents = ctx.scheduler.step(
            model_output=noise_prediction,
            timestep=step.t_device,
            sample=ctx.latents,
            **ctx.extra_step_kwargs,
            return_dict=False,
        )[0]
