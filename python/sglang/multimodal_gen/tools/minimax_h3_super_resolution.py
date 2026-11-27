# SPDX-License-Identifier: Apache-2.0
"""Generate a coarse FL2VA clip and refine it with a spatial SR LoRA."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", required=True, type=Path)
    parser.add_argument(
        "--image", required=True, type=Path, help="Original first frame"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--stage1-video", type=Path, help="Reuse an existing coarse H3 MP4"
    )
    parser.add_argument("--stage1-short-edge", type=int, default=512)
    parser.add_argument(
        "--target-short-edge",
        type=int,
        default=1448,
        help="Short-edge ceiling before the training pixel budget and 32px alignment",
    )
    parser.add_argument("--stage1-steps", type=int, default=50)
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument(
        "--seconds",
        type=float,
        help="Default: 5s for generation, source duration for an existing clip",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "task": "fl2va",
        "prompt": args.prompt,
        "seed": args.seed,
        "conditions": [
            {
                "type": "image",
                "uri": str(args.image.resolve(strict=True)),
                "role": "keyframe",
                "frame_index": 0,
            }
        ],
        "save_output": True,
        "output_path": str(args.output_dir.resolve()),
        "quality": "lossless",
    }
    seconds = args.seconds
    if seconds is None and args.stage1_video is not None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            _validate_localized_media,
        )

        facts = _validate_localized_media(
            str(args.stage1_video.resolve(strict=True)), condition_type="video"
        )
        # H3 rounds its 15s request up to a full VAE window.
        seconds = min(15.0, facts["video_duration_seconds"])
    target = {
        "aspect_ratio": "auto",
        "duration_seconds": 5.0 if seconds is None else seconds,
    }
    adapter = str(args.lora_path.resolve(strict=True))
    with DiffGenerator.from_pretrained(
        model_path=args.model_path,
        model_variant=None
        if Path(args.model_path).name.lower() == "fl2va"
        else "fl2va",
        num_gpus=args.num_gpus,
        tp_size=args.num_gpus,
        attention_backend="torch_sdpa",
    ) as generator:
        source = args.stage1_video
        if source is None:
            coarse = generator.generate(
                dict(
                    **common,
                    target=dict(**target, short_edge=args.stage1_short_edge),
                    num_inference_steps=args.stage1_steps,
                    output_file_name="stage1.mp4",
                )
            )
            if coarse is None:
                raise RuntimeError("Coarse FL2VA generation failed")
            source = Path(coarse.output_file_path)
        generator.set_lora("refinement", adapter, target="transformer")
        result = generator.generate(
            dict(
                **common,
                target=dict(**target, short_edge=args.target_short_edge),
                refinement_video=str(source.resolve(strict=True)),
                num_inference_steps=args.refinement_steps,
                output_file_name="refined.mp4",
            )
        )
        if result is None:
            raise RuntimeError("FL2VA refinement failed")
        print(result.output_file_path)


if __name__ == "__main__":
    main()
