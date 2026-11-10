"""How pixel values are scaled before a pretrained backbone sees them.

Frames leave the data loader as floats in 0..1, and augmentation and ViViT's
temporal compressor work in that range. A pretrained backbone was trained on
one particular scaling, so each model applies it just before the backbone:

- 'zero_one': unchanged, 0..1. Every run before 2026-11-09 used this, which
  matched neither backbone's pretraining.
- 'minus_one_one': 2x - 1, so -1..1. Google's Scenic config for the ViViT
  Kinetics-400 checkpoints sets zero_centering.
- 'hf_vivit': 4x - 3, so -3..1. Hugging Face's ViViT image processor produces
  this with the checkpoint's preprocessor_config.json: it rescales by 1/127.5
  with offset (-1..1), then normalizes with mean 0.5 and std 0.5 on top.
- 'imagenet': per-channel (x - mean) / std with ImageNet's statistics, which
  torchvision's IMAGENET1K_V1 ResNet weights expect.

scripts/check_input_scaling.py compares the ViViT options on real clips.
Checkpoints record the scaling they were trained with, and ones written
before the setting existed count as 'zero_one'.
"""
import torch

INPUT_SCALINGS = ('zero_one', 'minus_one_one', 'hf_vivit', 'imagenet')
# What checkpoints from before the setting existed were trained on
LEGACY_SCALING = 'zero_one'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def check_input_scaling(name):
    """Return `name` if it is a known scaling, otherwise raise."""
    if name not in INPUT_SCALINGS:
        raise ValueError(f"input_scaling must be one of {INPUT_SCALINGS}, got {name!r}")
    return name


def scale_frames(x, name, channel_dim):
    """Scale 0..1 frames for a backbone. `channel_dim` is the RGB axis of `x`."""
    if name == 'zero_one':
        return x
    if name == 'minus_one_one':
        return x * 2.0 - 1.0
    if name == 'hf_vivit':
        return x * 4.0 - 3.0
    if name == 'imagenet':
        shape = [1] * x.dim()
        shape[channel_dim] = len(IMAGENET_MEAN)
        mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(shape)
        std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(shape)
        return (x - mean) / std
    raise ValueError(f"input_scaling must be one of {INPUT_SCALINGS}, got {name!r}")


def model_input_scaling(model):
    """The scaling a model (or its DataParallel wrapper) applies."""
    return getattr(getattr(model, 'module', model), 'input_scaling', LEGACY_SCALING)


def checkpoint_input_scaling(checkpoint):
    """The scaling a checkpoint was trained with."""
    return checkpoint.get('input_scaling', LEGACY_SCALING)
