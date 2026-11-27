# SPDX-License-Identifier: Apache-2.0
"""Spatial FL2VA refinement of a first-pass RGB video."""

import subprocess

import numpy as np
import torch
from sglang.multimodal_gen.runtime.utils.process import local_media_subprocess_env

from .packed_tokens import minimax_h3_patchify_video_latent

REFINEMENT_EXTRA = "minimax_h3_refinement"
T_THRESH = 0.5


def refinement_sigmas(steps, audio_shift=3.0):
    # t_prime is sampled uniformly in [0, t_thresh] during training. Here
    # num_inference_steps counts actual Euler updates on that interval.
    video = torch.linspace(T_THRESH, 0, steps + 1)
    # The cache trainer indexes the descending audio schedule with t_prime
    # itself. Its zero clean audio latent therefore gets noisier as video clears.
    base = 1 - (video * 999).round() / 1000
    audio = audio_shift * base / (1 + (audio_shift - 1) * base)
    model_sigma = (audio * 1000).bfloat16().float() / 1000
    return {"video": video.tolist(), "audio": model_sigma.tolist()}


def refinement_frames(path, size, frame_count, keyframes):
    width, height = size
    # The training cache decodes at 24 fps and scales uint8 RGB with ffmpeg.
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "2",
        "-filter_threads",
        "2",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-vf",
        f"fps=24,scale={width}:{height}:flags=bilinear",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    with subprocess.Popen(
        command, env=local_media_subprocess_env(), stdout=subprocess.PIPE
    ) as process:
        size = process.stdout.readinto(frames)
        if process.wait():
            raise RuntimeError(f"ffmpeg could not decode refinement source {path}")
    whole_frames, remainder = divmod(size, frames[0].nbytes)
    if remainder or not whole_frames:
        raise ValueError("Refinement source did not decode to complete RGB frames")
    if whole_frames < frame_count:
        # The cache producer repeats the last frame up to the VAE boundary.
        frames[whole_frames:] = frames[whole_frames - 1]
    for index, image in keyframes.items():
        frames[index] = np.asarray(image.convert("RGB"))
    return frames


def encode_refinement_rows(vae, frames, arch, *, keyframe=False):
    # VeOmni encodes the posterior mean for both the RGB source and the
    # separate first-frame anchor, with ImageNet input and channel-wise latents.
    device = next(vae.parameters()).device
    pixels = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0).div_(255)
    pixels = vae.processor.transform_tensor(pixels)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        moments = vae.tiled_encode(pixels) if keyframe else vae.encode_temporal(pixels)
    mean = moments.chunk(2, dim=1)[0].float()
    if keyframe:
        mean = mean[:, :, :1]
    offset = mean.new_tensor(arch.latents_mean).view(1, -1, 1, 1, 1)
    scale = mean.new_tensor(arch.latents_std).view_as(offset)
    normalized = ((mean - offset) / scale).bfloat16()
    return minimax_h3_patchify_video_latent(normalized, patch_size=[1, 2, 2]).cpu()


def refinement_audio(facts, frame_count):
    from .reference_encoding import _audio_resampler, _load_waveform

    sample_rate = 32000
    samples = round(frame_count * sample_rate / 24)
    audio = torch.zeros(2, samples)
    if facts["has_audio"]:
        waveform, rate = _load_waveform(
            facts["local_path"],
            source_sample_rate=facts["audio_sample_rate"],
            max_duration_seconds=frame_count / 24,
        )
        if rate != sample_rate:
            waveform = _audio_resampler(rate)(waveform)
        length = min(samples, waveform.shape[-1])
        audio[:, :length] = waveform[:, :length]
    return audio[:, None, :]
