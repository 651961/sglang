# SPDX-License-Identifier: Apache-2.0

"""Sampling parameters for Bernini and Bernini-R."""

import html
import re
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_T2V_A14B_SamplingParam,
)


BERNINI_R_GUIDANCE_MODES = frozenset(
    {
        "auto",
        "rv2v",
        "v2v",
        "v2v_chain",
        "t2v",
        "r2v_apg",
        "v2v_apg",
        "t2v_apg",
    }
)

BERNINI_GUIDANCE_MODES = frozenset(
    {"auto", "vae_txt_vit", "vae_txt_vit_wapg", "rv2v_wapg"}
)

BERNINI_R_SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": (
        "You are a helpful assistant specialized in video editing on content "
        "propagation."
    ),
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": (
        "You are a helpful assistant for editing. "
        "You may need to adjust the subject's action or position."
    ),
    "mv2v": (
        "You are a helpful assistant for editing. You might need to adjust the "
        "video's style, lighting, colors, textures, and the subject's pose or action."
    ),
}


def bernini_prompt_clean(text: str) -> str:
    """Apply the user-prompt normalization from the official renderer."""
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def resolve_bernini_task_type(
    requested_task: str,
    *,
    has_video: bool,
    has_edit_image: bool,
    has_reference_image: bool,
    num_frames: int,
) -> str:
    """Infer the official task prompt when a request does not name one."""
    if requested_task != "auto":
        return requested_task
    if has_video:
        return "rv2v" if has_edit_image or has_reference_image else "v2v"
    if has_edit_image:
        return "i2i" if num_frames == 1 else "i2v"
    if has_reference_image:
        return "r2i" if num_frames == 1 else "r2v"
    return "t2i" if num_frames == 1 else "t2v"


@dataclass
class BerniniRSamplingParams(Wan2_2_T2V_A14B_SamplingParam):
    """Request parameters matching the official Bernini-R inference defaults.

    ``image_path`` keeps SGLang's public input name. A scalar path (including
    the shared CLI's one-element list) is the single-image editing source,
    while a multi-element list is interpreted as reference images.
    ``reference_image_paths`` is the unambiguous way to pass one or more
    references and also permits an editing image plus additional references.
    """

    height: int = 480
    width: int = 848
    supported_resolutions: list[tuple[int, int]] | None = None
    guidance_scale_2: float | None = None
    flow_shift: float | None = 5.0

    # Official inference prefixes a task-specific system instruction directly
    # (without an inserted separator) before prompt cleaning/tokenization.
    task_type: str = "auto"
    system_prompt: str | None = None
    use_system_prompt: bool | None = True

    # Bernini-R runs all of its guidance branches inside one request rather
    # than through SGLang's generic CFG policy.
    guidance_mode: str = "auto"
    omega_vid: float = 1.25
    omega_img: float = 4.5
    omega_txt: float | None = None
    omega_scale: float = 0.8
    eta: float = 0.5
    norm_threshold: float | tuple[float, ...] | list[float] = 50.0
    momentum: float = 0.0

    max_image_size: int = 848
    reference_image_paths: str | list[str] | None = None

    # Official inference creates FP32 noise from a CPU generator. Bernini-R
    # shards its packed tokens inside the transformer, so frame-count adjustment
    # by GPU count is unnecessary and would change the requested 4k+1 length.
    adjust_frames: bool = False

    def __post_init__(self) -> None:
        # argparse and the image-edit HTTP endpoint represent --image-path as a
        # list. The official renderer's singular ``image`` input is distinct
        # from its ``images`` reference list for both image and video outputs;
        # callers that mean a reference should use reference_image_paths.
        if isinstance(self.image_path, list) and len(self.image_path) == 1:
            self.image_path = self.image_path[0]

        explicit_fields = set(getattr(self, "_explicit_fields", ()))
        if self.omega_txt is None or (
            "guidance_scale" in explicit_fields and "omega_txt" not in explicit_fields
        ):
            # guidance_scale is the standard CLI/OpenAI alias. A renderer-native
            # omega_txt supplied alongside it takes precedence.
            self.omega_txt = float(self.guidance_scale)

        super().__post_init__()

        if self.guidance_mode not in BERNINI_R_GUIDANCE_MODES:
            supported = ", ".join(sorted(BERNINI_R_GUIDANCE_MODES))
            raise ValueError(
                f"Unsupported Bernini-R guidance_mode {self.guidance_mode!r}; "
                f"expected one of: {supported}"
            )
        if not isinstance(self.task_type, str) or not self.task_type:
            raise ValueError("task_type must be a non-empty string")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string or None")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("Bernini-R requires negative_prompt to be a string")
        if self.num_outputs_per_prompt != 1:
            raise ValueError("Bernini-R currently supports one output per prompt")
        if self.max_image_size <= 0:
            raise ValueError("max_image_size must be positive")

    def _set_output_file_name(self) -> None:
        # The renderer uses the same pipeline for images and videos.
        self.data_type = DataType.IMAGE if self.num_frames == 1 else DataType.VIDEO
        super()._set_output_file_name()

    def apply_request_extra(self, req) -> None:
        super().apply_request_extra(req)
        task_type = resolve_bernini_task_type(
            self.task_type,
            has_video=bool(self.video_path),
            has_edit_image=isinstance(self.image_path, str),
            has_reference_image=(
                isinstance(self.image_path, list) or bool(self.reference_image_paths)
            ),
            num_frames=self.num_frames,
        )
        req.extra["bernini_task_type"] = task_type
        if isinstance(req.prompt, str):
            req.prompt = bernini_prompt_clean(req.prompt)
        if self.use_system_prompt is not False:
            prefix = self.system_prompt or BERNINI_R_SYSTEM_PROMPTS.get(
                task_type, BERNINI_R_SYSTEM_PROMPTS["default"]
            )
            if isinstance(req.prompt, str):
                req.prompt = prefix + req.prompt

        # Every Bernini route evaluates a negative-text branch, even when the
        # numerical guidance scales are <= 1. Do not let generic CFG gating
        # skip negative-prompt encoding in that case.
        req.do_classifier_free_guidance = True
        # Req has a generic scheduler ``eta`` field, so explicitly propagate
        # the Bernini APG value instead of letting Req's 0.0 default shadow it.
        req.eta = self.eta

    @classmethod
    def get_cli_args(cls, args):
        cli_args = super().get_cli_args(args)
        # Keep the public ``--task-type`` spelling while using a distinct
        # argparse destination. Otherwise ServerArgs would mistake renderer
        # task prompts (for example ``rv2v``) for PipelineConfig.task_type.
        bernini_task_type = getattr(args, "bernini_task_type", None)
        if bernini_task_type is not None:
            cli_args["task_type"] = bernini_task_type
        # The shared CLI exposes --guidance-scale and --image-path. Map those
        # onto Bernini's text-guidance name and scalar edit-image semantics.
        if "guidance_scale" in cli_args and "omega_txt" not in cli_args:
            cli_args["omega_txt"] = cli_args["guidance_scale"]
        image_paths = cli_args.get("image_path")
        if isinstance(image_paths, list) and len(image_paths) == 1:
            cli_args["image_path"] = image_paths[0]
        return cli_args


@dataclass
class BerniniSamplingParams(BerniniRSamplingParams):
    """Bernini planner + renderer inference parameters."""

    num_inference_steps: int = 50
    guidance_mode: str = "auto"
    omega_vid: float = 1.0
    omega_img: float = 1.0
    omega_tgt: float = 0.5
    omega_scale: float = 1.0
    planning_step: int = 25
    vit_denoising_step: int = 5
    vit_txt_cfg: float = 1.2
    vit_img_cfg: float = 1.0
    system_prompt: str | None = ""
    max_image_size: int = 842
    max_sequence_length: int | None = 512

    def _apply_task_defaults(self) -> None:
        explicit = getattr(self, "_explicit_fields", None)
        if explicit is None:
            # The framework constructs a temporary user-params object before it
            # records which fields were explicitly supplied. Applying presets
            # during that first pass would erase those values.
            return
        explicit = set(explicit)
        task = resolve_bernini_task_type(
            self.task_type,
            has_video=bool(self.video_path),
            has_edit_image=isinstance(self.image_path, str),
            has_reference_image=(
                isinstance(self.image_path, list) or bool(self.reference_image_paths)
            ),
            num_frames=self.num_frames,
        )
        editing = task in {"i2i", "i2v", "v2v", "rv2v"}
        reference = task in {"r2i", "r2v", "rv2v"}
        defaults = {
            "num_inference_steps": 50,
            "omega_vid": 1.0,
            "omega_img": 1.0,
            "omega_tgt": 0.5,
            "omega_scale": 1.0,
            "max_image_size": 848 if task in {"v2v", "rv2v"} else 842,
            "guidance_mode": "rv2v_wapg" if task == "rv2v" else "vae_txt_vit_wapg",
        }
        if editing or reference:
            defaults.update(num_inference_steps=40, omega_vid=1.25)
        if editing:
            defaults.update(omega_img=1.25, omega_scale=0.75)
        if reference:
            defaults.update(omega_img=4.5, omega_tgt=1.5, omega_scale=0.8)
        if task == "rv2v":
            defaults.update(omega_vid=0.75, omega_img=3.0, omega_scale=0.75)
        if task in {"t2i", "i2i", "r2i"}:
            defaults.update(num_frames=1, height=512, width=512)
        for name, value in defaults.items():
            if name not in explicit:
                setattr(self, name, value)

    def __post_init__(self) -> None:
        if isinstance(self.image_path, list) and len(self.image_path) == 1:
            self.image_path = self.image_path[0]
        self._apply_task_defaults()
        explicit = set(getattr(self, "_explicit_fields", ()))
        if self.omega_txt is None or (
            "guidance_scale" in explicit and "omega_txt" not in explicit
        ):
            self.omega_txt = float(self.guidance_scale)

        Wan2_2_T2V_A14B_SamplingParam.__post_init__(self)
        if self.guidance_mode not in BERNINI_GUIDANCE_MODES:
            supported = ", ".join(sorted(BERNINI_GUIDANCE_MODES))
            raise ValueError(
                f"Unsupported Bernini guidance_mode {self.guidance_mode!r}; "
                f"expected one of: {supported}"
            )
        if self.planning_step <= 0 or self.vit_denoising_step <= 0:
            raise ValueError(
                "Bernini planning and visual denoising steps must be positive"
            )
        if not isinstance(self.task_type, str) or not self.task_type:
            raise ValueError("task_type must be a non-empty string")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string or None")
        if self.num_outputs_per_prompt != 1:
            raise ValueError("Bernini currently supports one output per prompt")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("Bernini requires negative_prompt to be a string")
        if self.max_image_size <= 0:
            raise ValueError("max_image_size must be positive")

    def apply_request_extra(self, req) -> None:
        Wan2_2_T2V_A14B_SamplingParam.apply_request_extra(self, req)
        task = resolve_bernini_task_type(
            self.task_type,
            has_video=bool(self.video_path),
            has_edit_image=isinstance(self.image_path, str),
            has_reference_image=(
                isinstance(self.image_path, list) or bool(self.reference_image_paths)
            ),
            num_frames=self.num_frames,
        )
        raw_prompt = (
            bernini_prompt_clean(req.prompt)
            if isinstance(req.prompt, str)
            else req.prompt
        )
        req.extra["bernini_task_type"] = task
        req.extra["bernini_raw_prompt"] = raw_prompt
        if isinstance(raw_prompt, str):
            prefix = self.system_prompt or ""
            req.prompt = bernini_prompt_clean(prefix + raw_prompt)
        req.do_classifier_free_guidance = True
        req.eta = self.eta
