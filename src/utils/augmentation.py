"""Random augmentation of training clips.

Each clip gets one random draw, applied to every frame alike, so the signing
stays consistent over time. Clips are never mirrored, since that would swap
the signer's dominant hand.

Random numbers come from torch, which DataLoader seeds separately in every
worker. Draws from numpy's global generator could repeat across workers.
"""
import math

import torch
import torch.nn.functional as F

# Segments the time warp divides a clip into
TIME_WARP_SEGMENTS = 3


def check_augmentation(config):
    """Reject augmentation settings that are out of range, before any clip loads."""
    low, high = config.augment_crop_scale
    if not 0 < low <= high <= 1:
        raise ValueError(f"augment_crop_scale must satisfy 0 < low <= high <= 1, got {config.augment_crop_scale}")
    low, high = config.augment_crop_ratio
    if not 0 < low <= high:
        raise ValueError(f"augment_crop_ratio must satisfy 0 < low <= high, got {config.augment_crop_ratio}")
    for name in ('augment_brightness', 'augment_contrast'):
        if not 0 <= getattr(config, name) < 1:
            raise ValueError(f"{name} must be in [0, 1), got {getattr(config, name)}")
    # Past 0.5 neighbouring segment boundaries could cross, running time backwards
    if not 0 <= config.augment_time_warp < 0.5:
        raise ValueError(f"augment_time_warp must be in [0, 0.5), got {config.augment_time_warp}")


def _uniform(low, high):
    return low + (high - low) * torch.rand(()).item()


def random_crop_box(height, width, scale, ratio):
    """A crop covering `scale` of the frame area at aspect `ratio`, at a random position."""
    area = _uniform(*scale) * height * width
    aspect = math.exp(_uniform(math.log(ratio[0]), math.log(ratio[1])))
    crop_w = min(width, max(1, round(math.sqrt(area * aspect))))
    crop_h = min(height, max(1, round(math.sqrt(area / aspect))))
    top = torch.randint(0, height - crop_h + 1, ()).item()
    left = torch.randint(0, width - crop_w + 1, ()).item()
    return top, left, crop_h, crop_w


def time_warp_indices(length, strength):
    """Frame indices that replay a clip with its speed varied along the way.

    The clip is cut into segments whose inner boundaries move by up to
    `strength` of a segment, so frames repeat where the signing slows down and
    are skipped where it speeds up. The first and last frames stay in place.
    """
    segments = TIME_WARP_SEGMENTS
    # Where each segment boundary of the output falls in the input, as a
    # share of the clip
    knots = [0.0]
    for k in range(1, segments):
        knots.append(k / segments + _uniform(-strength, strength) / segments)
    knots.append(1.0)

    indices = []
    for i in range(length):
        position = i / (length - 1) * segments
        k = min(int(position), segments - 1)
        source = knots[k] + (position - k) * (knots[k + 1] - knots[k])
        indices.append(round(source * (length - 1)))
    return torch.tensor(indices, dtype=torch.long)


def augment_clip(video, config):
    """Augment one (C, T, H, W) clip with values in [0, 1]."""
    _, frames, height, width = video.shape

    if config.augment_time_warp > 0 and frames > 1:
        video = video[:, time_warp_indices(frames, config.augment_time_warp)]

    top, left, crop_h, crop_w = random_crop_box(
        height, width, config.augment_crop_scale, config.augment_crop_ratio)
    if (crop_h, crop_w) != (height, width):
        crop = video[:, :, top:top + crop_h, left:left + crop_w]
        # Frames as the batch dimension, resized back to the model's input size
        video = F.interpolate(crop.permute(1, 0, 2, 3), size=(height, width),
                              mode='bilinear', align_corners=False).permute(1, 0, 2, 3)

    video = video * (1 + _uniform(-config.augment_brightness, config.augment_brightness))
    mean = video.mean()
    video = (video - mean) * (1 + _uniform(-config.augment_contrast, config.augment_contrast)) + mean
    return video.clamp(0, 1).contiguous()
