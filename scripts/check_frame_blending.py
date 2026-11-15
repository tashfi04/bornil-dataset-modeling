"""Whether blending frames before ViViT costs the pretrained model anything.

The temporal compressor averages neighbouring frames into each frame ViViT
sees, and ViViT was pretrained on single sharp frames. This passes the same
clips through the complete pretrained model, Kinetics classifier included,
with three kinds of input. Nothing is trained or saved.

- blended: the compressor, at its starting weights, reduces the clip to twice
  the classifier's frame count, as it does in training, and every second
  frame is used.
- sharp: the one real frame at the centre of each of those blended frames, so
  the two inputs cover the same moments and differ only in the blending.
- consecutive: neighbouring frames from the middle of the clip, at the spacing
  they were loaded with. This is closest to how ViViT was pretrained, but it
  covers only part of the clip.

The classifier takes half as many frames as our model feeds ViViT, which is
why every second blended frame is used. Clips longer than num_frames were
already thinned when they were cached, so for those "consecutive" frames are
further apart than in the recording.

    python scripts/check_frame_blending.py --config subset --csv path/to/subset.csv
"""
import os
import sys
import json
import math
import argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import VivitForVideoClassification

from src.models.vivit_ctc_hf import TemporalCompressor, DEFAULT_INPUT_SCALING
from src.utils.environment import check_environment
from src.utils.frame_cache import cache_path, read_cached_frames
from src.utils.input_scaling import scale_frames

VARIANTS = ('blended', 'sharp', 'consecutive')
# "sign language interpreting" in the standard alphabetical Kinetics-400 order
SIGNING_CLASS = 303


def load_variants(path, compressor, num_frames, size):
    """One cached clip as {variant: (num_frames, C, H, W) in 0..1}, and its frame count."""
    frames = read_cached_frames(path)
    if frames[0].shape[:2] != (size, size):
        frames = [cv2.resize(frame, (size, size)) for frame in frames]
    clip = torch.from_numpy(np.stack(frames)).float().div(255.0)   # (T, H, W, C), RGB
    clip = clip.permute(3, 0, 1, 2).unsqueeze(0)                   # (1, C, T, H, W)
    total = clip.size(2)

    with torch.no_grad():
        blended = compressor(clip)[0][:, ::2]                      # (C, num_frames, H, W)

    # Blended frame j of the compressor's output averages loaded frames around
    # (j + 0.5) * total / slots; every second j is kept
    slots = 2 * num_frames
    centres = [min(total - 1, int((2 * i + 0.5) * total / slots)) for i in range(num_frames)]
    sharp = clip[0][:, centres]

    start = max(0, (total - num_frames) // 2)
    consecutive = clip[0][:, [min(total - 1, start + i) for i in range(num_frames)]]

    variants = {'blended': blended, 'sharp': sharp, 'consecutive': consecutive}
    return {name: v.permute(1, 0, 2, 3).contiguous() for name, v in variants.items()}, total


def classify(model, clips, scaling, device):
    """Softmax over the Kinetics classes for a list of (T, C, H, W) clips."""
    batch = scale_frames(torch.stack(clips).to(device), scaling, channel_dim=2)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
        logits = model(pixel_values=batch).logits
    return logits.float().softmax(dim=-1).cpu()


def main():
    parser = argparse.ArgumentParser(description="Compare blended and sharp frames on the pretrained ViViT")
    parser.add_argument('--config', choices=['subset', 'test', 'prod', 'overfit'], default='subset')
    parser.add_argument('--csv', default=None,
                        help="Dataset CSV (defaults to the config's csv_path)")
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--clips', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    check_environment()

    if args.config == 'subset':
        from configs.subset_vivit_ctc_config import config
    elif args.config == 'test':
        from configs.test_vivit_ctc_config import config
    elif args.config == 'overfit':
        from configs.overfit_vivit_ctc_config import config
    else:
        from configs.vivit_ctc_config import config
    if args.csv:
        config.csv_path = args.csv

    cache_root = getattr(config, 'cached_frames_path', None)
    if not cache_root:
        sys.exit(f"The {args.config} config has no frame cache (cached_frames_path); "
                 f"this check reads clips from one.")

    df = pd.read_csv(config.csv_path)
    with open(config.train_val_test_split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)
    rows = df[df['recording'].map(split) == args.split]
    paths = []
    for _, row in rows.iterrows():
        path = cache_path(cache_root, row['chunk_path'], row['recording'])
        if os.path.exists(path):
            paths.append(path)
        if len(paths) == args.clips:
            break
    if not paths:
        sys.exit(f"No {args.split} clip of {config.csv_path} is in the cache at {cache_root}.")

    device = config.device
    model_name = getattr(config, 'vivit_model_name', 'google/vivit-b-16x2-kinetics400')
    model = VivitForVideoClassification.from_pretrained(model_name).to(device).eval()
    num_frames, height, width = model.config.video_size
    if height != width:
        sys.exit(f"{model_name} expects {height}x{width} frames; this check assumes square ones.")
    num_classes = model.config.num_labels
    scaling = getattr(config, 'input_scaling', None) or DEFAULT_INPUT_SCALING

    # A batch at a time, so only a few clips are held in memory
    compressor = TemporalCompressor(2 * num_frames)
    probs = {name: [] for name in VARIANTS}
    totals = []
    for start in range(0, len(paths), args.batch_size):
        batch = {name: [] for name in VARIANTS}
        for path in paths[start:start + args.batch_size]:
            variants, total = load_variants(path, compressor, num_frames, height)
            totals.append(total)
            for name in VARIANTS:
                batch[name].append(variants[name])
        for name in VARIANTS:
            probs[name].append(classify(model, batch[name], scaling, device))
    probs = {name: torch.cat(parts) for name, parts in probs.items()}

    blend = sum(max(1.0, total / (2 * num_frames)) for total in totals) / len(totals)
    thinned = sum(1 for total in totals if total >= config.num_frames)
    print(f"{len(paths)} {args.split} clips, {num_frames} frames each at {height}x{width}, input "
          f"scaling {scaling}, through the full {model_name} with its {num_classes}-class head")
    print(f"Loaded frames per clip: {min(totals)} to {max(totals)}; about {blend:.1f} blended into "
          f"each frame on average; {thinned} clips were cached at the {config.num_frames}-frame "
          f"limit, so they may have been thinned\n")

    max_entropy = math.log(num_classes)
    print(f"{'input':<12} {'mean top-1 prob':>16} {'mean entropy':>13} {'distinct top-1':>15} "
          f"{'top-1 is signing':>17} {'mean prob of signing':>21}")
    for name in VARIANTS:
        p = probs[name]
        best, best_index = p.max(dim=1)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
        signing = int((best_index == SIGNING_CLASS).sum())
        print(f"{name:<12} {best.mean().item():>16.3f} {entropy.mean().item() / max_entropy:>12.0%} "
              f"{len(Counter(best_index.tolist())):>15} {f'{signing}/{len(paths)}':>17} "
              f"{p[:, SIGNING_CLASS].mean().item():>21.3f}")
    print(f"(signing = class {SIGNING_CLASS}, 'sign language interpreting'; entropy is a share "
          f"of the maximum, lower means surer)")

    print("\nClip by clip, probability of signing:")
    for a, b in (('sharp', 'blended'), ('consecutive', 'blended'), ('consecutive', 'sharp')):
        pa, pb = probs[a][:, SIGNING_CLASS], probs[b][:, SIGNING_CLASS]
        higher = int((pa > pb).sum())
        print(f"  {a} higher than {b} in {higher}/{len(paths)} clips, "
              f"mean difference {(pa - pb).mean().item():+.3f}")

    print("\n=== Reading it ===")
    print("If 'sharp' clearly beats 'blended', averaging frames is costing the")
    print("pretrained model something, and picking real frames is worth trying in")
    print("training. If they are close, the blending is not the problem. This measures")
    print("how well the model recognises the activity, which is only a stand-in for")
    print("how well it can tell signs apart.")


if __name__ == "__main__":
    main()
