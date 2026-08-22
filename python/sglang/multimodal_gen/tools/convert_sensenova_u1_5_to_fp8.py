# SPDX-License-Identifier: Apache-2.0
"""Convert a SenseNova-U1.5 source checkpoint to serialized channel-wise FP8.

The output matches the weight quantization used by online ``--quantization fp8``
on NVIDIA GPUs with the CUTLASS FP8 path, while retaining the original HF tensor
names and layouts expected by the SenseNova loader.

Example:
    python -m sglang.multimodal_gen.tools.convert_sensenova_u1_5_to_fp8 \
        --model-dir /models/SenseNova-U1.5-8B-MoT \
        --save-dir /models/SenseNova-U1.5-8B-MoT-FP8 \
        --device cuda

Load the converted directory without passing ``--quantization fp8``. Its
``config.json`` selects the serialized FP8 path automatically.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min
FP8_EPS = 1e-10
SUPPORTED_SOURCE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
SUPPORTED_SOURCE_DTYPE_NAMES = {"F16", "BF16", "F32"}

_SENSENOVA_LINEAR_WEIGHT_RE = re.compile(
    r"^language_model\.model\.layers\.\d+\."
    r"(?:self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj|q_proj_mot_gen|k_proj_mot_gen|"
    r"v_proj_mot_gen|o_proj_mot_gen)|"
    r"(?:mlp|mlp_mot_gen)\.(?:gate_proj|up_proj|down_proj))\.weight$"
)

_ATTENTION_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_proj_mot_gen",
    "k_proj_mot_gen",
    "v_proj_mot_gen",
    "o_proj_mot_gen",
)
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def is_sensenova_fp8_weight(name: str) -> bool:
    return _SENSENOVA_LINEAR_WEIGHT_RE.fullmatch(name) is not None


def _expected_quantized_weights(config: dict[str, Any]) -> set[str]:
    llm_config = config.get("llm_config")
    if not isinstance(llm_config, dict) or "num_hidden_layers" not in llm_config:
        raise ValueError(
            "config.json is not a SenseNova-U1.5 config: missing "
            "llm_config.num_hidden_layers"
        )

    names = set()
    for layer_idx in range(int(llm_config["num_hidden_layers"])):
        prefix = f"language_model.model.layers.{layer_idx}"
        names.update(
            f"{prefix}.self_attn.{projection}.weight"
            for projection in _ATTENTION_PROJECTIONS
        )
        for mlp_name in ("mlp", "mlp_mot_gen"):
            names.update(
                f"{prefix}.{mlp_name}.{projection}.weight"
                for projection in _MLP_PROJECTIONS
            )
    return names


def channel_fp8(
    weight: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"FP8 linear weight must be 2D, got shape {weight.shape}")
    if weight.dtype not in SUPPORTED_SOURCE_DTYPES:
        raise ValueError(
            "FP8 linear weight must be an unquantized FP16, BF16, or FP32 "
            f"tensor, got {weight.dtype}"
        )

    # The native U1.5 loader materializes every quantized LinearBase in BF16
    # before online FP8 post-processing, including source checkpoint tensors
    # stored as FP32. Preserve that rounding step for an equivalent export.
    work = weight.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    work_fp32 = work.float()
    scale = work_fp32.abs().amax(dim=1, keepdim=True).clamp_min(FP8_EPS) / FP8_MAX
    qweight = (work_fp32 / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(FP8_DTYPE)
    return qweight.cpu().contiguous(), scale.cpu().contiguous()


def _load_and_validate_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing SenseNova config: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("quantization_config") is not None:
        raise ValueError(
            "The input config already contains quantization_config; use the original "
            "unquantized SenseNova-U1.5 checkpoint."
        )
    return config


def _inventory_weights(weight_files: list[Path]) -> set[str]:
    all_names: set[str] = set()
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in all_names:
                    raise ValueError(f"Duplicate tensor {name!r} in input checkpoint")
                if is_sensenova_fp8_weight(name):
                    dtype = handle.get_slice(name).get_dtype()
                    if dtype not in SUPPORTED_SOURCE_DTYPE_NAMES:
                        raise ValueError(
                            f"Target tensor {name!r} has source dtype {dtype}; expected "
                            "an unquantized FP16, BF16, or FP32 checkpoint"
                        )
                all_names.add(name)
    return all_names


def _copy_non_weight_files(model_dir: Path, save_dir: Path) -> None:
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    for source in model_dir.iterdir():
        if source.name == "config.json":
            continue
        if source.is_file() and (
            source.suffix in weight_suffixes
            or source.name.endswith(".safetensors.index.json")
        ):
            continue
        destination = save_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def convert_sensenova_u1_5_to_fp8(
    model_dir: str | Path,
    save_dir: str | Path,
    *,
    device: str = "auto",
) -> None:
    model_dir = Path(model_dir).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve()
    if model_dir == save_dir:
        raise ValueError("--model-dir and --save-dir must be different directories")
    if save_dir.is_relative_to(model_dir):
        raise ValueError("--save-dir must not be inside --model-dir")
    if "sensenova-u1.5" not in str(save_dir).lower():
        raise ValueError(
            "--save-dir must contain 'SenseNova-U1.5' so SGLang's local-model "
            "detector selects the native U1.5 pipeline"
        )
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory does not exist: {model_dir}")
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    config = _load_and_validate_config(model_dir)
    expected_quantized = _expected_quantized_weights(config)
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")
    all_names = _inventory_weights(weight_files)
    missing = sorted(expected_quantized - all_names)
    unexpected = sorted(name for name in all_names if is_sensenova_fp8_weight(name))
    unexpected = sorted(set(unexpected) - expected_quantized)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)} expected weights, e.g. {missing[:3]}"
            )
        if unexpected:
            details.append(
                f"found {len(unexpected)} unexpected layer weights, e.g. {unexpected[:3]}"
            )
        raise ValueError("Invalid SenseNova-U1.5 checkpoint: " + "; ".join(details))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_device = torch.device(device)
    if quant_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    _copy_non_weight_files(model_dir, save_dir)
    output_weight_map: dict[str, str] = {}
    total_size = 0
    converted_count = 0

    for weight_file in tqdm(weight_files, desc="Converting checkpoint shards"):
        output_tensors: dict[str, torch.Tensor] = {}
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if is_sensenova_fp8_weight(name):
                    qweight, scale = channel_fp8(tensor, quant_device)
                    output_tensors[name] = qweight
                    output_tensors[name.removesuffix(".weight") + ".weight_scale"] = (
                        scale
                    )
                    converted_count += 1
                    del qweight, scale
                else:
                    output_tensors[name] = tensor.contiguous()

        output_path = save_dir / weight_file.name
        save_file(output_tensors, output_path, metadata={"format": "pt"})
        for name, tensor in output_tensors.items():
            output_weight_map[name] = weight_file.name
            total_size += tensor.numel() * tensor.element_size()
        del output_tensors
        gc.collect()
        if quant_device.type == "cuda":
            torch.cuda.empty_cache()

    if converted_count != len(expected_quantized):
        raise RuntimeError(
            f"Converted {converted_count} weights, expected {len(expected_quantized)}"
        )

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_scale_strategy": "channel",
    }
    config["quantization_config"] = quantization_config
    with (save_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    with (save_dir / "model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(index, file, indent=2, sort_keys=True)
        file.write("\n")

    logger.info(
        "Converted %d SenseNova-U1.5 linear weights to channel-wise FP8 in %s",
        converted_count,
        save_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Original source model")
    parser.add_argument("--save-dir", required=True, help="Empty output directory")
    parser.add_argument(
        "--device",
        default="auto",
        help="Quantization device, for example cuda, cuda:0, or cpu (default: auto)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert_sensenova_u1_5_to_fp8(
        args.model_dir,
        args.save_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
