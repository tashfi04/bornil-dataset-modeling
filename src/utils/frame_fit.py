"""How a decoded video frame is fitted to the model's square input.

The recordings come from phones and laptops, so they are a mix of wide
(1920x1080, 1280x720) and tall (1080x1920) frames.

- 'stretch_smooth': resize straight to the target, shrinking by area
  averaging so fine detail is averaged instead of skipped. Wide frames come
  out squeezed sideways to 0.56 of their width and tall ones stretched to
  1.78, so the same hand shape looks different depending on the device.
- 'stretch': the same resize with OpenCV's default filter, which at these
  shrink factors (about 8x) reads only a few of the source pixels. Every run
  before 2026-11-15 used this.
- 'pad': scale the frame down until it fits, keeping its shape, and fill the
  rest with mid-grey. Shapes are right, but it does not work with the
  pretrained ViViT: with 40% of the patches flat grey, the layer-7 features
  barely change over time (0.19 against 0.69 for 'stretch', pretrained, share
  of variance from change over time), and the overfit check stayed blank until
  epoch 35.

Shared by the data loader and scripts/preprocess_cache_frames.py, so cached
frames match what the loader would produce on the fly.
"""
import cv2
import numpy as np

FRAME_FITS = ('stretch_smooth', 'stretch', 'pad')
DEFAULT_FRAME_FIT = 'stretch_smooth'
# What caches and runs from before the setting existed used
LEGACY_FRAME_FIT = 'stretch'

# Mid-grey is zero once frames are scaled to -1..1, the value a zero-centred
# backbone treats as "nothing here"
PAD_VALUE = 127


def check_frame_fit(name):
    """Return `name` if it is a known fit, otherwise raise."""
    if name not in FRAME_FITS:
        raise ValueError(f"frame_fit must be one of {FRAME_FITS}, got {name!r}")
    return name


def fitted_size(height, width, target_height, target_width):
    """(height, width) of a frame scaled to fit inside the target, shape kept."""
    scale = min(target_height / height, target_width / width)
    return (max(1, min(target_height, round(height * scale))),
            max(1, min(target_width, round(width * scale))))


def fit_frame(frame, size, fit=DEFAULT_FRAME_FIT):
    """Fit one HxWx3 uint8 frame to `size`, given as (height, width)."""
    target_height, target_width = size
    if fit == 'stretch':
        return cv2.resize(frame, (target_width, target_height))
    if fit not in FRAME_FITS:
        raise ValueError(f"frame_fit must be one of {FRAME_FITS}, got {fit!r}")

    height, width = frame.shape[:2]
    if fit == 'stretch_smooth':
        shrinking = target_height < height or target_width < width
        return cv2.resize(frame, (target_width, target_height),
                          interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)

    new_height, new_width = fitted_size(height, width, target_height, target_width)
    shrinking = new_height < height or new_width < width
    resized = cv2.resize(frame, (new_width, new_height),
                         interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)

    canvas = np.full((target_height, target_width, 3), PAD_VALUE, dtype=frame.dtype)
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas
