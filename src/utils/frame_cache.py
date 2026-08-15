"""Container format for cached, pre-sampled video frames.

The source clips are webm with missing metadata and must be decoded
sequentially, so reading them every epoch is expensive. Caching the sampled and
resized frames as JPEG reduces that to a handful of JPEG decodes per clip.

One file per recording. Layout (little-endian, no padding):

    magic    4s          b'BFC1'
    count    u16         number of frames
    height   u16
    width    u16
    lengths  u32 * count byte length of each JPEG blob
    blobs                concatenated JPEG bytes
"""
import os
import struct

import cv2
import numpy as np

MAGIC = b'BFC1'
_HEADER = '<4sHHH'
_HEADER_SIZE = struct.calcsize(_HEADER)
CACHE_SUFFIX = '.bfc'


def encode_frames(frames, quality=90):
    """Encode a list of HxWx3 uint8 RGB frames into cache bytes."""
    if len(frames) == 0:
        raise ValueError("Cannot encode an empty frame list")

    blobs = []
    for frame in frames:
        ok, buf = cv2.imencode(
            '.jpg',
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        blobs.append(buf.tobytes())

    height, width = frames[0].shape[:2]
    header = struct.pack(_HEADER, MAGIC, len(blobs), height, width)
    lengths = struct.pack(f'<{len(blobs)}I', *(len(b) for b in blobs))
    return header + lengths + b''.join(blobs)


def decode_frames(data):
    """Decode cache bytes back into a list of HxWx3 uint8 RGB frames."""
    magic, count, height, width = struct.unpack_from(_HEADER, data, 0)
    if magic != MAGIC:
        raise ValueError(f"Not a frame-cache file (magic={magic!r})")

    offset = _HEADER_SIZE
    lengths = struct.unpack_from(f'<{count}I', data, offset)
    offset += 4 * count

    frames = []
    for length in lengths:
        blob = np.frombuffer(data, dtype=np.uint8, count=length, offset=offset)
        offset += length
        img = cv2.imdecode(blob, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Corrupt JPEG blob in frame cache")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if len(frames) != count:
        raise ValueError(f"Expected {count} frames, decoded {len(frames)}")
    return frames


def read_cached_frames(path):
    with open(path, 'rb') as f:
        return decode_frames(f.read())


def write_cached_frames(path, frames, quality=90):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(encode_frames(frames, quality=quality))


def cache_path(cache_root, chunk_path, recording):
    """Mirror the dataset's chunk_path/recording layout inside the cache root."""
    stem = os.path.splitext(recording)[0]
    return os.path.join(cache_root, chunk_path, stem + CACHE_SUFFIX)
