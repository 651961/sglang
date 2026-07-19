# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Bernini semantic planner built around Qwen2.5-VL."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    ModulateProjection,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import (
    Qwen2_5_VLForConditionalGeneration,
)


def _modulate(
    hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return hidden_states * (1 + scale) + shift


class _VisualTokenResBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = MLP(hidden_size, hidden_size, hidden_size, act_type="silu")
        self.adaLN_modulation = ModulateProjection(
            hidden_size, factor=3, act_layer="silu"
        )

    def forward(
        self, hidden_states: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        residual = self.mlp(_modulate(self.in_ln(hidden_states), shift, scale))
        return hidden_states + gate * residual


class _VisualTokenFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size)
        self.adaLN_modulation = ModulateProjection(
            hidden_size, factor=2, act_layer="silu"
        )

    def forward(
        self, hidden_states: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(hidden_states), shift, scale))


class _VisualTokenMLP(nn.Module):
    def __init__(
        self, input_size: int, condition_size: int, hidden_size: int, depth: int
    ) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedder(hidden_size, act_layer="silu")
        self.cond_embed = nn.Linear(condition_size, hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.res_blocks = nn.ModuleList(
            [_VisualTokenResBlock(hidden_size) for _ in range(depth)]
        )
        self.final_layer = _VisualTokenFinalLayer(hidden_size, input_size)

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.input_proj(x)
        modulation = self.time_embed(timestep) + self.cond_embed(condition)
        for block in self.res_blocks:
            hidden_states = block(hidden_states, modulation)
        return self.final_layer(hidden_states, modulation)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        half = x[: len(x) // 2]
        prediction = self(torch.cat((half, half)), timestep, condition)
        conditional, unconditional = prediction.chunk(2, dim=0)
        guided = unconditional + cfg_scale * (conditional - unconditional)
        return torch.cat((guided, guided), dim=0)

    def forward_with_txt_img_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        txt_cfg_scale: float,
        img_cfg_scale: float,
    ) -> torch.Tensor:
        third = x[: len(x) // 3]
        prediction = self(torch.cat((third, third, third)), timestep, condition)
        conditional, unconditional, image_conditional = prediction.chunk(3, dim=0)
        guided = (
            unconditional
            + img_cfg_scale * (image_conditional - unconditional)
            + txt_cfg_scale * (conditional - image_conditional)
        )
        return torch.cat((guided, guided, guided), dim=0)


class _FlowMatchScheduler:
    def __init__(self, shift: float = 2.0, extra_one_step: bool = True) -> None:
        self.shift = shift
        self.extra_one_step = extra_one_step

    def set_timesteps(
        self, steps: int, *, device: torch.device, dtype: torch.dtype
    ) -> None:
        count = steps + int(self.extra_one_step)
        sigmas = torch.linspace(1.0, 0.003 / 1.002, count, device=device, dtype=dtype)
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * 1000

    def step(
        self, prediction: torch.Tensor, index: int, sample: torch.Tensor
    ) -> torch.Tensor:
        next_sigma = self.sigmas[index + 1] if index + 1 < len(self.sigmas) else 0
        return sample + prediction * (next_sigma - self.sigmas[index])


class _VisualTokenDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_size: int,
        hidden_size: int,
        depth: int,
        shift: float,
        extra_one_step: bool,
    ) -> None:
        super().__init__()
        self.net = _VisualTokenMLP(channels, condition_size, hidden_size, depth)
        self.scheduler = _FlowMatchScheduler(shift, extra_one_step)
        self.channels = channels

    def sample(
        self,
        condition: torch.Tensor,
        *,
        txt_cfg: float,
        image_cfg: float | None,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device, dtype = condition.device, condition.dtype

        def make_noise(count: int) -> torch.Tensor:
            return torch.randn(
                count, self.channels, generator=generator, device="cpu"
            ).to(device)

        if image_cfg is not None and txt_cfg > 1:
            sample = make_noise(condition.shape[0] // 3).repeat(3, 1)
            forward = self.net.forward_with_txt_img_cfg
            forward_kwargs = {"txt_cfg_scale": txt_cfg, "img_cfg_scale": image_cfg}
        elif txt_cfg > 1:
            sample = make_noise(condition.shape[0] // 2).repeat(2, 1)
            forward = self.net.forward_with_cfg
            forward_kwargs = {"cfg_scale": txt_cfg}
        else:
            sample = make_noise(condition.shape[0])
            forward = self.net.forward
            forward_kwargs = {}

        self.scheduler.set_timesteps(steps, device=device, dtype=dtype)
        sample = sample.to(dtype)
        for index, timestep in enumerate(self.scheduler.timesteps):
            prediction = forward(
                sample,
                timestep.reshape(1).to(dtype),
                condition,
                **forward_kwargs,
            )
            sample = self.scheduler.step(prediction, index, sample)
        return sample


class BerniniPlannerModel(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL planner, connector, and visual-token flow decoder."""

    def __init__(self, config: Qwen2_5VLConfig) -> None:
        config.enable_image_understanding = True
        config.arch_config.text_config.use_explicit_attention_mask = True
        architecture = config.arch_config
        super().__init__(config)
        hidden_size = architecture.text_config.hidden_size
        connector_config = architecture.bernini_connector_cfg
        visual_config = architecture.bernini_clip_diff_cfg
        self.mask_tokens = nn.Parameter(
            torch.empty(1, architecture.bernini_num_mask_token, hidden_size)
        )
        gen_dim = connector_config.get("out_dim_for_gen", 4096)
        vit_dim = connector_config.get("out_dim_for_vit", hidden_size)
        self.connector = nn.ModuleDict(
            {
                "proj_gen": nn.Sequential(
                    nn.Linear(hidden_size, gen_dim),
                    nn.GELU(),
                    RMSNorm(gen_dim),
                    nn.Linear(gen_dim, gen_dim),
                ),
                "pred_vit": nn.Sequential(
                    nn.Linear(hidden_size, vit_dim),
                    nn.GELU(),
                    nn.Linear(vit_dim, vit_dim),
                    RMSNorm(vit_dim),
                    nn.Linear(vit_dim, vit_dim),
                ),
            }
        )
        self.vit_decoder = _VisualTokenDecoder(
            channels=visual_config.get("target_channels", hidden_size),
            condition_size=visual_config.get("z_channels", hidden_size),
            hidden_size=visual_config.get("width", 4096),
            depth=visual_config.get("depth", 16),
            shift=visual_config.get("shift", 2.0),
            extra_one_step=visual_config.get("extra_one_step", True),
        )

    def hidden_states(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_masks = {
            "full_attention": attention_mask,
            "sliding_attention": attention_mask,
        }
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-2]

    def visual_features(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.model.get_image_features(
            pixel_values.to(self.device), grid_thw.to(self.device)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded_direct: set[str] = set()

        def qwen_weights():
            for name, tensor in weights:
                if name.startswith("mllm."):
                    yield name.removeprefix("mllm."), tensor
                    continue
                name = name.replace(".mlp.0.", ".mlp.fc_in.")
                name = name.replace(".mlp.2.", ".mlp.fc_out.")
                name = name.replace(".adaLN_modulation.1.", ".adaLN_modulation.linear.")
                parameter = parameters[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, tensor.to(parameter.dtype))
                loaded_direct.add(name)

        loaded_qwen = super().load_weights(qwen_weights())
        return loaded_direct | loaded_qwen


EntryClass = BerniniPlannerModel
